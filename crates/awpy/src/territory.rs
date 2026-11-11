//! Territory: map control by travel time.
//!
//! [`reachability_control`](crate::map_control::reachability_control) gives
//! every area to whichever side is nearer in walking distance between area
//! centres. Territory asks *how long* each player needs to step into each
//! area, and only counts space a side can reach within a time horizon:
//!
//! - Paths run through *portals*, the stretches of edge two connected areas
//!   share, rather than through area centres. Areas are convex, so walking
//!   straight from one portal to the next is always possible, and paths come
//!   out close to the straight lines players run. Centre-to-centre paths
//!   zigzag across the small nav polygons and read about a quarter too long.
//! - Each player runs at their own speed.
//! - Dropping off a ledge adds part of the fall, jumping up a ledge adds a
//!   jump, and ladders are climbed at ladder speed. Ramps and stairs walk
//!   along their slope.
//! - Burning ground is impassable. Each burn is a disc tested against the
//!   area's polygon, not just its centre, so a flame at the edge of a large
//!   area still closes it.
//! - Areas no one reaches within [`TerritoryParams::horizon`] seconds are
//!   neutral; an area both sides reach within
//!   [`TerritoryParams::contest_margin`] seconds of each other is contested.
//!
//! The mesh is prepared once with [`Territory::new`]; [`Territory::control`]
//! then labels one snapshot.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::map_control::Team;
use crate::nav::{Nav, NavArea};

type Vec3 = [f32; 3];

/// Where a runner starts: their position and the area it lies in.
type Start = (Vec3, usize);

/// How far a position may be from every area and still be placed on the
/// nearest one, in Hammer units.
const SNAP_DISTANCE: f32 = 256.0;

/// Two edges belong to the same portal when they run within this many units
/// of each other in the plane.
const PORTAL_TOLERANCE: f32 = 4.0;

/// How far from the end of a shared edge a player can cross it.
const PLAYER_WIDTH: f32 = 16.0;

/// Tunable parts of the travel-time model. Times are in seconds, lengths in
/// Hammer units.
#[derive(Clone, Copy, Debug)]
pub struct TerritoryParams {
    /// Space no player reaches within this many seconds is neutral (default
    /// `f32::INFINITY`, i.e. no limit).
    pub horizon: f32,
    /// Two sides reaching an area within this many seconds of each other
    /// contest it (default `0.6`).
    pub contest_margin: f32,
    /// A ledge higher than this needs a jump (default `20`; stairs step 18).
    pub jump_height: f32,
    /// Extra time to jump up a ledge (default `0.35`).
    pub jump_time: f32,
    /// A drop deeper than this is a fall rather than a step (default `64`).
    pub fall_height: f32,
    /// Gravity for falls, units/s² (default `800`, CS2's `sv_gravity`).
    pub gravity: f32,
    /// Share of a fall's time that adds to the trip (default `0.5`): players
    /// keep moving forward while they fall.
    pub fall_weight: f32,
    /// Ladder climbing speed (default `150`).
    pub ladder_speed: f32,
    /// Time to get on and off a ladder (default `0.4`).
    pub ladder_mount: f32,
}

impl Default for TerritoryParams {
    fn default() -> Self {
        Self {
            horizon: f32::INFINITY,
            contest_margin: 0.6,
            jump_height: 20.0,
            jump_time: 0.35,
            fall_height: 64.0,
            gravity: 800.0,
            fall_weight: 0.5,
            ladder_speed: 150.0,
            ladder_mount: 0.4,
        }
    }
}

/// A living player: feet position, side and running speed (units/s).
#[derive(Clone, Copy, Debug)]
pub struct Runner {
    pub pos: Vec3,
    pub team: Team,
    pub speed: f32,
}

/// Burning ground: a disc of `radius` around `center`. A `NaN` height burns
/// every floor under the disc; otherwise only areas within 100 units of it.
#[derive(Clone, Copy, Debug)]
pub struct Burn {
    pub center: Vec3,
    pub radius: f32,
}

/// One snapshot's territory.
#[derive(Clone, Debug)]
pub struct TerritoryControl {
    /// Per area in nav order: bit 0 set when CT holds it, bit 1 when T does
    /// (both: contested, neither: neutral).
    pub labels: Vec<u8>,
    /// Size-weighted shares of the map: CT alone, T alone, contested, neutral.
    pub ct_fraction: f32,
    pub t_fraction: f32,
    pub contested_fraction: f32,
    pub neutral_fraction: f32,
}

/// A way from one area into a connected one.
#[derive(Clone, Copy, Debug)]
struct Portal {
    /// The area it leads into.
    to: u32,
    /// Where it is left from, on the first area.
    depart: Vec3,
    /// Where it is arrived at, on the area it leads into.
    arrive: Vec3,
    ladder: bool,
}

/// A navigation mesh prepared for travel-time queries.
#[derive(Clone, Debug)]
pub struct Territory {
    corners: Vec<Vec<Vec3>>,
    centroids: Vec<Vec3>,
    sizes: Vec<f32>,
    total_size: f32,
    bounds: Vec<[f32; 4]>,
    portals: Vec<Portal>,
    /// Portals out of each area: `portals[out_starts[a]..out_starts[a + 1]]`.
    out_starts: Vec<u32>,
    /// Portals into each area, as indices into `portals`.
    into: Vec<Vec<u32>>,
}

#[derive(PartialEq)]
struct Visit {
    time: f32,
    node: u32,
}

impl Eq for Visit {}

impl Ord for Visit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .total_cmp(&self.time)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for Visit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn distance(a: Vec3, b: Vec3) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Where to cross the stretch of edge `from` shares with `to` in the plane:
/// its middle, and on a stretch wide enough also a player's width in from
/// each end, so paths can cut corners. Each crossing is a point on `from`'s
/// edge and the point under or over it on `to`'s edge. Empty when no edges
/// run together (a jump across a gap, say).
fn crossings(from: &NavArea, to: &NavArea) -> Vec<(Vec3, Vec3)> {
    let (n, m) = (from.corners.len(), to.corners.len());
    let mut best: Option<(f32, Vec3, Vec3, Vec3, Vec3, f32, f32)> = None;
    for i in 0..n {
        let (a0, a1) = (from.corners[i], from.corners[(i + 1) % n]);
        let (dx, dy) = (a1[0] - a0[0], a1[1] - a0[1]);
        let len2 = dx * dx + dy * dy;
        if len2 <= f32::EPSILON {
            continue;
        }
        let len = len2.sqrt();
        for j in 0..m {
            let (b0, b1) = (to.corners[j], to.corners[(j + 1) % m]);
            // Both of b's ends must lie on a's line.
            let off = |p: Vec3| ((p[0] - a0[0]) * dy - (p[1] - a0[1]) * dx).abs() / len;
            if off(b0) > PORTAL_TOLERANCE || off(b1) > PORTAL_TOLERANCE {
                continue;
            }
            let along = |p: Vec3| ((p[0] - a0[0]) * dx + (p[1] - a0[1]) * dy) / len2;
            let (t0, t1) = (along(b0), along(b1));
            let (lo, hi) = (t0.min(t1).max(0.0), t0.max(t1).min(1.0));
            let overlap = (hi - lo) * len;
            if overlap > 1.0 && best.is_none_or(|(o, ..)| o < overlap) {
                best = Some((overlap, a0, a1, b0, b1, lo, hi));
            }
        }
    }
    let Some((overlap, a0, a1, b0, b1, lo, hi)) = best else {
        return Vec::new();
    };
    let (t0, t1) = {
        let (dx, dy) = (a1[0] - a0[0], a1[1] - a0[1]);
        let len2 = dx * dx + dy * dy;
        let along = |p: Vec3| ((p[0] - a0[0]) * dx + (p[1] - a0[1]) * dy) / len2;
        (along(b0), along(b1))
    };
    let mut at = vec![(lo + hi) / 2.0];
    if overlap > 3.0 * PLAYER_WIDTH {
        let inset = PLAYER_WIDTH / (overlap / (hi - lo));
        at.extend([lo + inset, hi - inset]);
    }
    at.into_iter()
        .map(|t| {
            let depart = lerp(a0, a1, t);
            // The same point on b's edge, for its height.
            let s = if (t1 - t0).abs() <= f32::EPSILON {
                0.0
            } else {
                ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
            };
            (depart, [depart[0], depart[1], lerp(b0, b1, s)[2]])
        })
        .collect()
}

/// The pair of corners, one from each area, nearest each other in the plane.
fn nearest_corners(from: &NavArea, to: &NavArea) -> (Vec3, Vec3) {
    let mut best = (f32::INFINITY, [0.0; 3], [0.0; 3]);
    for a in &from.corners {
        for b in &to.corners {
            let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
            if d < best.0 {
                best = (d, *a, *b);
            }
        }
    }
    (best.1, best.2)
}

/// Squared distance from `(x, y)` to the segment `a`–`b` in the plane.
fn segment_distance_sq(x: f32, y: f32, a: Vec3, b: Vec3) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx * dx + dy * dy;
    let t = if len <= f32::EPSILON {
        0.0
    } else {
        (((x - a[0]) * dx + (y - a[1]) * dy) / len).clamp(0.0, 1.0)
    };
    let (px, py) = (a[0] + dx * t - x, a[1] + dy * t - y);
    px * px + py * py
}

fn contains_xy(corners: &[Vec3], x: f32, y: f32) -> bool {
    let n = corners.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (corners[i][0], corners[i][1]);
        let (xj, yj) = (corners[j][0], corners[j][1]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl Territory {
    /// Prepare `nav`: centres, sizes, and a portal for every connection and for
    /// both directions of every ladder.
    pub fn new(nav: &Nav) -> Territory {
        let index: HashMap<u32, u32> = nav
            .areas
            .iter()
            .enumerate()
            .map(|(i, area)| (area.area_id, i as u32))
            .collect();
        let centroids: Vec<Vec3> = nav.areas.iter().map(NavArea::centroid).collect();

        // Areas at either end of each ladder.
        let mut ladders: HashMap<u32, Vec<u32>> = HashMap::new();
        for (i, area) in nav.areas.iter().enumerate() {
            for ladder in area.ladders_above.iter().chain(&area.ladders_below) {
                ladders.entry(*ladder).or_default().push(i as u32);
            }
        }

        let mut out_starts = Vec::with_capacity(nav.areas.len() + 1);
        let mut portals = Vec::new();
        for (i, area) in nav.areas.iter().enumerate() {
            out_starts.push(portals.len() as u32);
            let mut seen: Vec<u32> = Vec::new();
            for id in &area.connections {
                let Some(&to) = index.get(id) else {
                    continue;
                };
                if seen.contains(&to) || to as usize == i {
                    continue;
                }
                seen.push(to);
                let other = &nav.areas[to as usize];
                let mut points = crossings(area, other);
                if points.is_empty() {
                    points.push(nearest_corners(area, other));
                }
                for (depart, arrive) in points {
                    portals.push(Portal {
                        to,
                        depart,
                        arrive,
                        ladder: false,
                    });
                }
            }
            for ladder in area.ladders_above.iter().chain(&area.ladders_below) {
                for &to in ladders.get(ladder).into_iter().flatten() {
                    // Only the far end: areas at the same end share a height.
                    if seen.contains(&to)
                        || to as usize == i
                        || (centroids[to as usize][2] - centroids[i][2]).abs() <= 32.0
                    {
                        continue;
                    }
                    seen.push(to);
                    let (depart, arrive) = nearest_corners(area, &nav.areas[to as usize]);
                    portals.push(Portal {
                        to,
                        depart,
                        arrive,
                        ladder: true,
                    });
                }
            }
        }
        out_starts.push(portals.len() as u32);
        let mut into = vec![Vec::new(); nav.areas.len()];
        for (p, portal) in portals.iter().enumerate() {
            into[portal.to as usize].push(p as u32);
        }

        let sizes: Vec<f32> = nav.areas.iter().map(NavArea::size).collect();
        let bounds = nav
            .areas
            .iter()
            .map(|area| {
                area.corners.iter().fold(
                    [
                        f32::INFINITY,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY,
                    ],
                    |b, c| {
                        [
                            b[0].min(c[0]),
                            b[1].min(c[1]),
                            b[2].max(c[0]),
                            b[3].max(c[1]),
                        ]
                    },
                )
            })
            .collect();
        Territory {
            corners: nav.areas.iter().map(|area| area.corners.clone()).collect(),
            centroids,
            total_size: sizes.iter().sum(),
            sizes,
            bounds,
            portals,
            out_starts,
            into,
        }
    }

    /// Number of areas.
    pub fn area_count(&self) -> usize {
        self.centroids.len()
    }

    /// The area a position stands on: among areas containing it in the plane,
    /// the one whose centre height is closest. A position over no area is
    /// placed on the nearest centre within 256 units, else `None`.
    pub fn locate(&self, pos: Vec3) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, corners) in self.corners.iter().enumerate() {
            let b = self.bounds[i];
            if pos[0] < b[0] || pos[0] > b[2] || pos[1] < b[1] || pos[1] > b[3] {
                continue;
            }
            if contains_xy(corners, pos[0], pos[1]) {
                let dz = (self.centroids[i][2] - pos[2]).abs();
                if best.is_none_or(|(_, d)| dz < d) {
                    best = Some((i, dz));
                }
            }
        }
        if let Some((i, _)) = best {
            return Some(i);
        }
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, distance(*c, pos)))
            .filter(|(_, d)| *d <= SNAP_DISTANCE)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// Seconds to pass through `portal` once standing at its departure point.
    fn crossing_time(portal: &Portal, params: &TerritoryParams) -> f32 {
        let step = portal.arrive[2] - portal.depart[2];
        if portal.ladder {
            return params.ladder_mount + step.abs() / params.ladder_speed;
        }
        if step < -params.fall_height {
            return params.fall_weight * (2.0 * -step / params.gravity).sqrt();
        }
        if step > params.jump_height {
            return params.jump_time;
        }
        0.0
    }

    /// Is area `i` under burning ground?
    fn burning(&self, i: usize, burns: &[Burn]) -> bool {
        let b = self.bounds[i];
        let corners = &self.corners[i];
        burns.iter().any(|burn| {
            let [x, y, z] = burn.center;
            let r = burn.radius;
            if x + r < b[0] || x - r > b[2] || y + r < b[1] || y - r > b[3] {
                return false;
            }
            if !z.is_nan() && (self.centroids[i][2] - z).abs() > 100.0 {
                return false;
            }
            if contains_xy(corners, x, y) {
                return true;
            }
            let n = corners.len();
            (0..n).any(|k| segment_distance_sq(x, y, corners[k], corners[(k + 1) % n]) <= r * r)
        })
    }

    /// Time for the nearest of `starts` (each a position and the area it
    /// stands in) to reach every portal at `speed`, never entering a blocked
    /// area or going past the horizon. A portal's time is when its far side is
    /// reached. One flood from several runners at the same speed gives each
    /// portal the time of whichever arrives first.
    fn flood(
        &self,
        starts: &[Start],
        speed: f32,
        blocked: &[bool],
        params: &TerritoryParams,
        times: &mut [f32],
    ) {
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Visit>, times: &mut [f32], node: usize, time: f32| {
            if time <= params.horizon && time < times[node] {
                times[node] = time;
                heap.push(Visit {
                    time,
                    node: node as u32,
                });
            }
        };
        let out = |area: usize| self.out_starts[area] as usize..self.out_starts[area + 1] as usize;
        for &(from, start) in starts {
            for p in out(start) {
                let portal = &self.portals[p];
                if !blocked[portal.to as usize] {
                    let time =
                        distance(from, portal.depart) / speed + Self::crossing_time(portal, params);
                    push(&mut heap, times, p, time);
                }
            }
        }
        while let Some(Visit { time, node }) = heap.pop() {
            if time > times[node as usize] {
                continue;
            }
            let here = self.portals[node as usize];
            for p in out(here.to as usize) {
                let portal = &self.portals[p];
                if blocked[portal.to as usize] {
                    continue;
                }
                let walk = if here.ladder {
                    // Off a ladder: the height is already climbed.
                    (portal.depart[0] - here.arrive[0]).hypot(portal.depart[1] - here.arrive[1])
                } else {
                    distance(here.arrive, portal.depart)
                };
                push(
                    &mut heap,
                    times,
                    p,
                    time + walk / speed + Self::crossing_time(portal, params),
                );
            }
        }
    }

    /// Seconds for a runner at `from` to get to `to` at `speed`, or `None`
    /// when either end is off the mesh or there is no way within the horizon.
    pub fn travel_time(
        &self,
        from: Vec3,
        to: Vec3,
        speed: f32,
        burns: &[Burn],
        params: &TerritoryParams,
    ) -> Option<f32> {
        let (start, target) = (self.locate(from)?, self.locate(to)?);
        if start == target {
            return Some(distance(from, to) / speed);
        }
        let blocked: Vec<bool> = (0..self.area_count())
            .map(|i| self.burning(i, burns))
            .collect();
        if blocked[start] {
            return None;
        }
        let mut times = vec![f32::INFINITY; self.portals.len()];
        self.flood(&[(from, start)], speed, &blocked, params, &mut times);
        self.into[target]
            .iter()
            .map(|&p| times[p as usize] + distance(self.portals[p as usize].arrive, to) / speed)
            .filter(|time| time.is_finite())
            .min_by(f32::total_cmp)
    }

    /// Label one snapshot. Runners with a non-positive speed or off the mesh
    /// are ignored.
    pub fn control(
        &self,
        runners: &[Runner],
        burns: &[Burn],
        params: &TerritoryParams,
    ) -> TerritoryControl {
        let n = self.area_count();
        let blocked: Vec<bool> = (0..n).map(|i| self.burning(i, burns)).collect();
        let mut ct = vec![f32::INFINITY; n];
        let mut t = vec![f32::INFINITY; n];
        let mut scratch = vec![f32::INFINITY; self.portals.len()];
        // One flood per side and speed: runners at the same speed share it.
        let mut groups: Vec<(Team, f32, Vec<Start>)> = Vec::new();
        for runner in runners {
            if runner.speed.is_nan() || runner.speed <= 0.0 {
                continue;
            }
            let Some(start) = self.locate(runner.pos) else {
                continue;
            };
            if blocked[start] {
                continue;
            }
            match groups
                .iter_mut()
                .find(|(team, speed, _)| *team == runner.team && *speed == runner.speed)
            {
                Some(group) => group.2.push((runner.pos, start)),
                None => groups.push((runner.team, runner.speed, vec![(runner.pos, start)])),
            }
        }
        for (team, speed, starts) in &groups {
            scratch.fill(f32::INFINITY);
            self.flood(starts, *speed, &blocked, params, &mut scratch);
            let best = match team {
                Team::Ct => &mut ct,
                Team::T => &mut t,
            };
            for &(_, start) in starts {
                best[start] = 0.0;
            }
            for (p, time) in scratch.iter().enumerate() {
                let to = self.portals[p].to as usize;
                if *time < best[to] {
                    best[to] = *time;
                }
            }
        }

        let mut labels = vec![0u8; n];
        let mut shares = [0.0f32; 4];
        for i in 0..n {
            let (c, tt) = (ct[i], t[i]);
            let label = match (c.is_finite(), tt.is_finite()) {
                (true, true) if (c - tt).abs() <= params.contest_margin => 3,
                (true, true) if c < tt => 1,
                (true, true) => 2,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 0,
            };
            labels[i] = label;
            let slot = match label {
                1 => 0,
                2 => 1,
                3 => 2,
                _ => 3,
            };
            shares[slot] += self.sizes[i];
        }
        let total = if self.total_size > 0.0 {
            self.total_size
        } else {
            1.0
        };
        TerritoryControl {
            labels,
            ct_fraction: shares[0] / total,
            t_fraction: shares[1] / total,
            contested_fraction: shares[2] / total,
            neutral_fraction: shares[3] / total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 100 × 100 square area at `(x, y)`, floor height `z`, linked to `conns`.
    fn square(id: u32, x: f32, y: f32, z: f32, conns: &[u32]) -> NavArea {
        NavArea {
            area_id: id,
            hull_index: 0,
            dynamic_attribute_flags: 0,
            corners: vec![
                [x, y, z],
                [x + 100.0, y, z],
                [x + 100.0, y + 100.0, z],
                [x, y + 100.0, z],
            ],
            connections: conns.to_vec(),
            ladders_above: vec![],
            ladders_below: vec![],
        }
    }

    fn runner(x: f32, y: f32, z: f32, team: Team) -> Runner {
        Runner {
            pos: [x, y, z],
            team,
            speed: 250.0,
        }
    }

    /// Five squares in a row along x, each linked to its neighbours.
    fn corridor() -> Nav {
        Nav::from_areas(
            (0..5u32)
                .map(|i| {
                    let mut conns = vec![];
                    if i > 0 {
                        conns.push(i);
                    }
                    if i < 4 {
                        conns.push(i + 2);
                    }
                    square(i + 1, i as f32 * 100.0, 0.0, 0.0, &conns)
                })
                .collect(),
        )
    }

    #[test]
    fn the_faster_side_takes_the_middle() {
        let territory = Territory::new(&corridor());
        let params = TerritoryParams {
            contest_margin: 0.05,
            ..TerritoryParams::default()
        };
        let even = territory.control(
            &[
                runner(50.0, 50.0, 0.0, Team::Ct),
                runner(450.0, 50.0, 0.0, Team::T),
            ],
            &[],
            &params,
        );
        assert_eq!(even.labels, vec![1, 1, 3, 2, 2]);

        // A T on a knife against a CT with a heavy weapon wins the middle.
        let slow_ct = Runner {
            speed: 150.0,
            ..runner(50.0, 50.0, 0.0, Team::Ct)
        };
        let uneven = territory.control(&[slow_ct, runner(450.0, 50.0, 0.0, Team::T)], &[], &params);
        assert_eq!(uneven.labels, vec![1, 1, 2, 2, 2]);
    }

    #[test]
    fn space_beyond_the_horizon_is_neutral() {
        let territory = Territory::new(&corridor());
        let params = TerritoryParams {
            horizon: 0.9,
            ..TerritoryParams::default()
        };
        // At 250 units/s the CT steps into the areas at 0.2, 0.6 and 1.0 s.
        let control = territory.control(&[runner(50.0, 50.0, 0.0, Team::Ct)], &[], &params);
        assert_eq!(control.labels, vec![1, 1, 1, 0, 0]);
        assert!((control.ct_fraction - 0.6).abs() < 1e-5);
        assert!((control.neutral_fraction - 0.4).abs() < 1e-5);
    }

    #[test]
    fn paths_cut_straight_through_portals() {
        // A 2 × 2 block of squares: going corner to corner crosses the shared
        // edges, never via the far centres.
        let nav = Nav::from_areas(vec![
            square(1, 0.0, 0.0, 0.0, &[2, 3]),
            square(2, 100.0, 0.0, 0.0, &[1, 4]),
            square(3, 0.0, 100.0, 0.0, &[1, 4]),
            square(4, 100.0, 100.0, 0.0, &[2, 3]),
        ]);
        let territory = Territory::new(&nav);
        let time = territory
            .travel_time(
                [20.0, 50.0, 0.0],
                [180.0, 50.0, 0.0],
                250.0,
                &[],
                &TerritoryParams::default(),
            )
            .unwrap();
        assert!((time - 160.0 / 250.0).abs() < 1e-4, "{time}");
    }

    #[test]
    fn paths_cut_corners_on_wide_edges() {
        let nav = Nav::from_areas(vec![
            square(1, 0.0, 0.0, 0.0, &[2, 3]),
            square(2, 100.0, 0.0, 0.0, &[1, 4]),
            square(3, 0.0, 100.0, 0.0, &[1, 4]),
            square(4, 100.0, 100.0, 0.0, &[2, 3]),
        ]);
        let territory = Territory::new(&nav);
        let time = territory
            .travel_time(
                [20.0, 20.0, 0.0],
                [180.0, 180.0, 0.0],
                250.0,
                &[],
                &TerritoryParams::default(),
            )
            .unwrap();
        // The straight line is 226 units; through the edge middles it is 241.
        assert!(time * 250.0 < 230.0, "{}", time * 250.0);
    }

    #[test]
    fn dropping_is_quick_and_climbing_needs_a_jump() {
        // Area 1 is a ledge 200 units above area 2; the drop is one-way.
        let nav = Nav::from_areas(vec![
            square(1, 0.0, 0.0, 200.0, &[2]),
            square(2, 100.0, 0.0, 0.0, &[]),
        ]);
        let territory = Territory::new(&nav);
        let params = TerritoryParams::default();
        let fall = Territory::crossing_time(&territory.portals[0], &params);
        assert_eq!(territory.portals.len(), 3);
        assert!(
            (fall - 0.5 * (2.0f32 * 200.0 / 800.0).sqrt()).abs() < 1e-4,
            "{fall}"
        );
        assert!(
            territory
                .travel_time([150.0, 50.0, 0.0], [50.0, 50.0, 200.0], 250.0, &[], &params)
                .is_none()
        );

        let ledge = Nav::from_areas(vec![
            square(1, 0.0, 0.0, 0.0, &[2]),
            square(2, 100.0, 0.0, 40.0, &[1]),
        ]);
        let territory = Territory::new(&ledge);
        let up = Territory::crossing_time(&territory.portals[0], &params);
        assert!((up - params.jump_time).abs() < 1e-4, "{up}");
    }

    #[test]
    fn ladders_join_their_ends() {
        let mut top = square(1, 0.0, 0.0, 300.0, &[]);
        let mut bottom = square(2, 0.0, 200.0, 0.0, &[]);
        top.ladders_above = vec![7];
        bottom.ladders_below = vec![7];
        let territory = Territory::new(&Nav::from_areas(vec![top, bottom]));
        let control = territory.control(
            &[runner(50.0, 250.0, 0.0, Team::T)],
            &[],
            &TerritoryParams::default(),
        );
        assert_eq!(control.labels, vec![2, 2]);
        let climb = Territory::crossing_time(&territory.portals[1], &TerritoryParams::default());
        assert!((climb - (0.4 + 300.0 / 150.0)).abs() < 1e-4, "{climb}");
    }

    #[test]
    fn a_flame_at_the_edge_closes_an_area_and_the_path_through_it() {
        let territory = Territory::new(&corridor());
        // Inside area 3 near its edge: 45 units from its centre, 50 from its neighbours.
        let burn = Burn {
            center: [250.0, 5.0, f32::NAN],
            radius: 20.0,
        };
        let control = territory.control(
            &[runner(50.0, 50.0, 0.0, Team::Ct)],
            &[burn],
            &TerritoryParams::default(),
        );
        assert_eq!(control.labels, vec![1, 1, 0, 0, 0]);

        // A burn on another floor leaves this one open.
        let above = Burn {
            center: [250.0, 50.0, 500.0],
            radius: 60.0,
        };
        let open = territory.control(
            &[runner(50.0, 50.0, 0.0, Team::Ct)],
            &[above],
            &TerritoryParams::default(),
        );
        assert_eq!(open.labels, vec![1; 5]);
    }

    #[test]
    fn travel_time_runs_the_corridor_in_a_straight_line() {
        let territory = Territory::new(&corridor());
        let params = TerritoryParams::default();
        let time = territory
            .travel_time([50.0, 50.0, 0.0], [450.0, 50.0, 0.0], 250.0, &[], &params)
            .unwrap();
        assert!((time - 1.6).abs() < 1e-4, "{time}");
        assert!(
            territory
                .travel_time([50.0, 900.0, 0.0], [450.0, 50.0, 0.0], 250.0, &[], &params)
                .is_none()
        );
    }

    #[test]
    fn players_just_off_the_mesh_are_placed_on_it() {
        let territory = Territory::new(&corridor());
        assert_eq!(territory.locate([50.0, 50.0, 0.0]), Some(0));
        assert_eq!(territory.locate([50.0, 150.0, 0.0]), Some(0));
        assert_eq!(territory.locate([50.0, 900.0, 0.0]), None);
    }
}
