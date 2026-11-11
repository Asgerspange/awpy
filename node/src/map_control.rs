//! Map control for Node.js: one navigation mesh, many snapshots per call.
//!
//! The caller already holds player positions (it sampled them while parsing),
//! so this takes positions instead of re-reading the demo. Samples are split
//! across threads, capped by `AWPY_TICK_SEGMENTS` like the demo decoder.
//!
//! [`NavMesh::territory`] is the model the parser uses: travel time per
//! player, a horizon, ladders, falls and burning ground (see
//! [`awpy::territory`]). [`NavMesh::reachability`] is awpy's plain
//! walking-distance model, kept for comparison. Loading a collision mesh with
//! [`NavMesh::load_sight`] adds what each side can see on top of territory.

use std::num::NonZeroUsize;
use std::path::Path;

use awpy::geometry::VisibilityMesh;
use awpy::map_control::{Observer, Occluder, Params, Team, raycast_control, reachability_control};
use awpy::nav::Nav;
use awpy::territory::{Burn, Runner, Territory, TerritoryParams};
use napi::bindgen_prelude::{Float32Array, Uint8Array, Uint32Array};
use napi_derive::napi;

use crate::js_error;

/// How far an off-mesh player may be from an area centroid and still be
/// placed on it, in Hammer units. Players standing on props or mid-jump are
/// often just outside every polygon.
const SNAP_DISTANCE: f32 = 256.0;

/// Default blocked radius around a burning inferno, in Hammer units (the value
/// awpy's Python `MapControlParams` uses).
const FIRE_RADIUS: f64 = 150.0;

/// Smoke cloud radius and the height of its centre above where it landed (the
/// values awpy's Python `MapControlParams` uses).
const SMOKE_RADIUS: f32 = 144.0;
const SMOKE_HEIGHT: f32 = 60.0;

/// Area outlines in nav order, for drawing the labels.
#[napi(object)]
pub struct NavOutlines {
    /// Corner count of each area.
    pub corners: Uint32Array,
    /// `[x, y, z]` of every corner, area after area.
    pub points: Float32Array,
}

/// Player positions and burning fires for a run of snapshots.
///
/// Sample `i` owns observer rows `offsets[i]..offsets[i + 1]` and fire rows
/// `fire_offsets[i]..fire_offsets[i + 1]`; both offset arrays hold one more
/// entry than there are samples.
#[napi(object)]
pub struct ControlInput {
    pub offsets: Uint32Array,
    /// `2` for T, `3` for CT. Other rows are ignored.
    pub team: Uint8Array,
    pub x: Float32Array,
    pub y: Float32Array,
    pub z: Float32Array,
    pub fire_offsets: Uint32Array,
    pub fire_x: Float32Array,
    pub fire_y: Float32Array,
    pub fire_z: Float32Array,
    /// Blocked radius around each fire. Defaults to 150.
    pub fire_radius: Option<f64>,
    /// Travel-distance band within which an area is contested. Defaults to 200.
    pub contest_margin: Option<f64>,
}

/// Like [`ControlInput`], with each player's speed and each burn's own radius.
/// A `NaN` burn height burns every floor under it. Smokes are only read when a
/// sight mesh is loaded and `sight` is set.
#[napi(object)]
pub struct TerritoryInput {
    pub offsets: Uint32Array,
    /// `2` for T, `3` for CT. Other rows are ignored.
    pub team: Uint8Array,
    pub x: Float32Array,
    pub y: Float32Array,
    pub z: Float32Array,
    /// Running speed of each player, units/s.
    pub speed: Float32Array,
    pub fire_offsets: Uint32Array,
    pub fire_x: Float32Array,
    pub fire_y: Float32Array,
    pub fire_z: Float32Array,
    pub fire_radius: Float32Array,
    /// Seconds beyond which space is neutral. Unlimited when absent.
    pub horizon: Option<f64>,
    /// Seconds within which two sides contest an area. Defaults to 0.6.
    pub contest_margin: Option<f64>,
    /// `0` or absent: territory only. `1`: an area the other side can see is
    /// contested. `2`: also, space no one reaches in time goes to the side that
    /// sees it.
    pub sight: Option<u32>,
    /// Crouched (`1`) per player, for eye height. Read with `sight`.
    pub crouched: Option<Uint8Array>,
    /// Blinded (`1`) per player; a blind player sees nothing. Read with `sight`.
    pub blind: Option<Uint8Array>,
    pub smoke_offsets: Option<Uint32Array>,
    pub smoke_x: Option<Float32Array>,
    pub smoke_y: Option<Float32Array>,
    /// Where each smoke landed; `NaN` puts it on the highest floor under it.
    pub smoke_z: Option<Float32Array>,
}

/// Moves to time: from one position to another at a speed.
#[napi(object)]
pub struct TravelInput {
    pub from_x: Float32Array,
    pub from_y: Float32Array,
    pub from_z: Float32Array,
    pub to_x: Float32Array,
    pub to_y: Float32Array,
    pub to_z: Float32Array,
    /// Running speed per move, units/s.
    pub speed: Float32Array,
    /// `true`: awpy's walking distance at the given speed. Otherwise
    /// travel time (falls, jumps, ladders).
    pub distance: Option<bool>,
}

/// Map control per sample.
#[napi(object)]
pub struct ControlColumns {
    /// CT, T, contested and neutral shares of the map for each sample, weighted
    /// by area size. Four values per sample.
    pub shares: Float32Array,
    /// Two bits per area in nav order (bit 0: CT holds it, bit 1: T holds it),
    /// four areas per byte, `row_bytes` bytes per sample.
    pub labels: Uint8Array,
    pub row_bytes: u32,
}

/// A navigation mesh loaded from an awpy `.nav` file.
#[napi]
pub struct NavMesh {
    nav: Nav,
    centroids: Vec<[f32; 3]>,
    territory: Territory,
    sight: Option<VisibilityMesh>,
}

/// One sample's inputs, copied out of the JavaScript arrays.
#[derive(Default)]
struct Sample {
    observers: Vec<Observer>,
    runners: Vec<Runner>,
    occluders: Vec<Occluder>,
    burns: Vec<Burn>,
    smokes: Vec<Occluder>,
}

fn thread_budget() -> usize {
    std::env::var("AWPY_TICK_SEGMENTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, NonZeroUsize::get))
}

fn check_offsets(offsets: &[u32], rows: usize, what: &str) -> napi::Result<()> {
    let ordered = offsets.windows(2).all(|pair| pair[0] <= pair[1]);
    let last = offsets.last().copied().unwrap_or(0) as usize;
    if offsets.is_empty() || offsets[0] != 0 || !ordered || last != rows {
        return Err(js_error(format!("bad {what} offsets")));
    }
    Ok(())
}

fn team_of(code: u8) -> Option<Team> {
    match code {
        3 => Some(Team::Ct),
        2 => Some(Team::T),
        _ => None,
    }
}

/// Size-weighted shares of each label code (CT, T, contested, neutral).
fn shares_of(labels: &[u8], sizes: &[f32]) -> [f32; 4] {
    let mut shares = [0.0f32; 4];
    for (label, size) in labels.iter().zip(sizes) {
        let slot = match label {
            1 => 0,
            2 => 1,
            3 => 2,
            _ => 3,
        };
        shares[slot] += size;
    }
    let total: f32 = shares.iter().sum();
    if total > 0.0 {
        for share in &mut shares {
            *share /= total;
        }
    }
    shares
}

/// Run `label` over every sample on up to `AWPY_TICK_SEGMENTS` threads and
/// pack the per-area labels two bits each.
fn run_samples<F>(samples: &[Sample], area_count: usize, label: F) -> napi::Result<ControlColumns>
where
    F: Fn(&Sample) -> ([f32; 4], Vec<u8>) + Sync,
{
    let row_bytes = area_count.div_ceil(4);
    let threads = thread_budget().min(samples.len().div_ceil(8)).max(1);
    let per_thread = samples.len().div_ceil(threads).max(1);
    let label = &label;
    let parts: Vec<(Vec<f32>, Vec<u8>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = samples
            .chunks(per_thread)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut shares = Vec::with_capacity(chunk.len() * 4);
                    let mut packed = vec![0u8; chunk.len() * row_bytes];
                    for (index, sample) in chunk.iter().enumerate() {
                        let (sample_shares, codes) = label(sample);
                        shares.extend_from_slice(&sample_shares);
                        let row = &mut packed[index * row_bytes..(index + 1) * row_bytes];
                        for (area, code) in codes.iter().enumerate() {
                            row[area / 4] |= (code & 3) << ((area % 4) * 2);
                        }
                    }
                    (shares, packed)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or_default())
            .collect()
    });
    let mut shares = Vec::with_capacity(samples.len() * 4);
    let mut labels = Vec::with_capacity(samples.len() * row_bytes);
    for (part_shares, part_labels) in parts {
        shares.extend(part_shares);
        labels.extend(part_labels);
    }
    if shares.len() != samples.len() * 4 {
        return Err(js_error("map control thread failed"));
    }
    Ok(ControlColumns {
        shares: Float32Array::new(shares),
        labels: Uint8Array::new(labels),
        row_bytes: u32::try_from(row_bytes).unwrap_or(u32::MAX),
    })
}

#[napi]
impl NavMesh {
    /// Load a `.nav` file.
    #[napi(constructor)]
    pub fn new(path: String) -> napi::Result<Self> {
        let nav = Nav::from_file(Path::new(&path)).map_err(js_error)?;
        let centroids = nav.areas.iter().map(awpy::nav::NavArea::centroid).collect();
        let territory = Territory::new(&nav);
        Ok(Self {
            nav,
            centroids,
            territory,
            sight: None,
        })
    }

    #[napi(getter)]
    pub fn area_count(&self) -> u32 {
        u32::try_from(self.nav.area_count()).unwrap_or(u32::MAX)
    }

    /// Load the map's collision mesh (an awpy `.mesh` file) so `territory` can
    /// add what each side sees.
    #[napi]
    pub fn load_sight(&mut self, path: String) -> napi::Result<()> {
        self.sight = Some(VisibilityMesh::from_file(Path::new(&path)).map_err(js_error)?);
        Ok(())
    }

    /// Index, in label order, of the area a position stands on, or `null` when
    /// it is more than 256 units from every area.
    #[napi]
    pub fn locate(&self, x: f64, y: f64, z: f64) -> Option<u32> {
        self.territory
            .locate([x as f32, y as f32, z as f32])
            .and_then(|index| u32::try_from(index).ok())
    }

    /// Seconds each move takes by the model, `NaN` when either end is off the
    /// mesh or there is no way through. For checking the model against how
    /// players really move.
    #[napi]
    pub fn travel_times(&self, input: TravelInput) -> napi::Result<Float32Array> {
        let count = input.from_x.len();
        if [
            input.from_y.len(),
            input.from_z.len(),
            input.to_x.len(),
            input.to_y.len(),
            input.to_z.len(),
            input.speed.len(),
        ]
        .iter()
        .any(|&len| len != count)
        {
            return Err(js_error("move columns differ in length"));
        }
        let params = TerritoryParams {
            horizon: 60.0,
            ..TerritoryParams::default()
        };
        let by_distance = input.distance.unwrap_or(false);
        let moves: Vec<([f32; 3], [f32; 3], f32)> = (0..count)
            .map(|i| {
                (
                    [input.from_x[i], input.from_y[i], input.from_z[i]],
                    [input.to_x[i], input.to_y[i], input.to_z[i]],
                    input.speed[i],
                )
            })
            .collect();
        let threads = thread_budget().min(count.div_ceil(64)).max(1);
        let per_thread = count.div_ceil(threads).max(1);
        let times: Vec<f32> = std::thread::scope(|scope| {
            let handles: Vec<_> = moves
                .chunks(per_thread)
                .map(|chunk| {
                    let params = &params;
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|(from, to, speed)| {
                                self.travel(*from, *to, *speed, by_distance, params)
                            })
                            .collect::<Vec<f32>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap_or_default())
                .collect()
        });
        if times.len() != count {
            return Err(js_error("travel thread failed"));
        }
        Ok(Float32Array::new(times))
    }

    /// Every area's polygon, in the order labels use.
    #[napi]
    pub fn outlines(&self) -> NavOutlines {
        let mut corners = Vec::with_capacity(self.nav.areas.len());
        let mut points = Vec::new();
        for area in &self.nav.areas {
            corners.push(u32::try_from(area.corners.len()).unwrap_or(0));
            for corner in &area.corners {
                points.extend_from_slice(corner);
            }
        }
        NavOutlines {
            corners: Uint32Array::new(corners),
            points: Float32Array::new(points),
        }
    }

    /// awpy's walking-distance model for each sample: an area belongs to the
    /// side that can walk to it first, a near-tie is contested, and fires
    /// block the ground under them.
    #[napi]
    pub fn reachability(&self, input: ControlInput) -> napi::Result<ControlColumns> {
        let rows = input.team.len();
        if input.x.len() != rows || input.y.len() != rows || input.z.len() != rows {
            return Err(js_error("observer columns differ in length"));
        }
        let fire_rows = input.fire_x.len();
        if input.fire_y.len() != fire_rows || input.fire_z.len() != fire_rows {
            return Err(js_error("fire columns differ in length"));
        }
        check_offsets(&input.offsets, rows, "observer")?;
        check_offsets(&input.fire_offsets, fire_rows, "fire")?;
        if input.fire_offsets.len() != input.offsets.len() {
            return Err(js_error("observer and fire sample counts differ"));
        }
        let radius = input.fire_radius.unwrap_or(FIRE_RADIUS) as f32;
        let samples: Vec<Sample> = input
            .offsets
            .windows(2)
            .zip(input.fire_offsets.windows(2))
            .map(|(players, fires)| Sample {
                observers: (players[0] as usize..players[1] as usize)
                    .filter_map(|row| {
                        let pos = [input.x[row], input.y[row], input.z[row]];
                        if pos.iter().any(|value| !value.is_finite()) {
                            return None;
                        }
                        Some(Observer {
                            pos: self.on_mesh(pos)?,
                            team: team_of(input.team[row])?,
                            crouched: false,
                            blind: false,
                            yaw: None,
                        })
                    })
                    .collect(),
                occluders: (fires[0] as usize..fires[1] as usize)
                    .map(|row| Occluder {
                        center: [input.fire_x[row], input.fire_y[row], input.fire_z[row]],
                        radius,
                    })
                    .collect(),
                ..Sample::default()
            })
            .collect();
        let params = Params {
            contest_margin: input
                .contest_margin
                .unwrap_or(Params::default().contest_margin),
            ..Params::default()
        };
        run_samples(&samples, self.nav.area_count(), |sample| {
            let control =
                reachability_control(&self.nav, &sample.observers, &sample.occluders, &params);
            let codes = control
                .areas
                .iter()
                .map(|area| u8::from(area.ct) | (u8::from(area.t) << 1))
                .collect();
            (
                [
                    control.ct_fraction,
                    control.t_fraction,
                    control.contested_fraction,
                    control.neutral_fraction,
                ],
                codes,
            )
        })
    }

    /// Travel-time territory for each sample (see [`awpy::territory`]), with
    /// what each side sees on top when a sight mesh is loaded and asked for.
    #[napi]
    pub fn territory(&self, input: TerritoryInput) -> napi::Result<ControlColumns> {
        let rows = input.team.len();
        if [
            input.x.len(),
            input.y.len(),
            input.z.len(),
            input.speed.len(),
        ]
        .iter()
        .any(|&len| len != rows)
        {
            return Err(js_error("observer columns differ in length"));
        }
        let fire_rows = input.fire_x.len();
        if [
            input.fire_y.len(),
            input.fire_z.len(),
            input.fire_radius.len(),
        ]
        .iter()
        .any(|&len| len != fire_rows)
        {
            return Err(js_error("fire columns differ in length"));
        }
        check_offsets(&input.offsets, rows, "observer")?;
        check_offsets(&input.fire_offsets, fire_rows, "fire")?;
        if input.fire_offsets.len() != input.offsets.len() {
            return Err(js_error("observer and fire sample counts differ"));
        }
        let sight_mode = input.sight.unwrap_or(0);
        let sight = if sight_mode > 0 {
            Some(
                self.sight
                    .as_ref()
                    .ok_or_else(|| js_error("no sight mesh loaded"))?,
            )
        } else {
            None
        };
        let flag = |column: &Option<Uint8Array>, row: usize| {
            column
                .as_ref()
                .and_then(|values| values.get(row))
                .is_some_and(|&v| v == 1)
        };
        let empty_offsets = vec![0u32; input.offsets.len()];
        let smoke_offsets: &[u32] = input.smoke_offsets.as_deref().unwrap_or(&empty_offsets);
        let smoke_rows = input.smoke_x.as_ref().map_or(0, |values| values.len());
        if sight.is_some() {
            check_offsets(smoke_offsets, smoke_rows, "smoke")?;
            if smoke_offsets.len() != input.offsets.len() {
                return Err(js_error("observer and smoke sample counts differ"));
            }
        }
        let smoke_value = |column: &Option<Float32Array>, row: usize| {
            column
                .as_ref()
                .and_then(|values| values.get(row).copied())
                .unwrap_or(f32::NAN)
        };

        let samples: Vec<Sample> = (0..input.offsets.len() - 1)
            .map(|index| {
                let players = input.offsets[index] as usize..input.offsets[index + 1] as usize;
                let fires =
                    input.fire_offsets[index] as usize..input.fire_offsets[index + 1] as usize;
                let mut sample = Sample::default();
                for row in players {
                    let pos = [input.x[row], input.y[row], input.z[row]];
                    let Some(team) = team_of(input.team[row]) else {
                        continue;
                    };
                    if pos.iter().any(|value| !value.is_finite()) {
                        continue;
                    }
                    sample.runners.push(Runner {
                        pos,
                        team,
                        speed: input.speed[row],
                    });
                    if sight.is_some() {
                        sample.observers.push(Observer {
                            pos,
                            team,
                            crouched: flag(&input.crouched, row),
                            blind: flag(&input.blind, row),
                            yaw: None,
                        });
                    }
                }
                sample.burns = fires
                    .map(|row| Burn {
                        center: [input.fire_x[row], input.fire_y[row], input.fire_z[row]],
                        radius: input.fire_radius[row],
                    })
                    .collect();
                if sight.is_some() {
                    sample.smokes = (smoke_offsets[index] as usize
                        ..smoke_offsets[index + 1] as usize)
                        .filter_map(|row| {
                            let (x, y) = (
                                smoke_value(&input.smoke_x, row),
                                smoke_value(&input.smoke_y, row),
                            );
                            let landed = smoke_value(&input.smoke_z, row);
                            let z = if landed.is_nan() {
                                self.top_floor(x, y)?
                            } else {
                                landed
                            };
                            Some(Occluder {
                                center: [x, y, z + SMOKE_HEIGHT],
                                radius: SMOKE_RADIUS,
                            })
                        })
                        .collect();
                }
                sample
            })
            .collect();

        let defaults = TerritoryParams::default();
        let params = TerritoryParams {
            horizon: input.horizon.map_or(defaults.horizon, |value| value as f32),
            contest_margin: input
                .contest_margin
                .map_or(defaults.contest_margin, |value| value as f32),
            ..defaults
        };
        let sizes: Vec<f32> = self
            .nav
            .areas
            .iter()
            .map(awpy::nav::NavArea::size)
            .collect();
        run_samples(&samples, self.nav.area_count(), |sample| {
            let control = self
                .territory
                .control(&sample.runners, &sample.burns, &params);
            let Some(mesh) = sight else {
                return (
                    [
                        control.ct_fraction,
                        control.t_fraction,
                        control.contested_fraction,
                        control.neutral_fraction,
                    ],
                    control.labels,
                );
            };
            let seen = raycast_control(
                &self.nav,
                mesh,
                &sample.observers,
                &sample.smokes,
                &Params::default(),
            );
            let codes: Vec<u8> = control
                .labels
                .iter()
                .zip(&seen.areas)
                .map(|(&label, area)| match (label, area.ct, area.t) {
                    (1, _, true) | (2, true, _) => 3,
                    (0, ct, t) if sight_mode >= 2 => u8::from(ct) | (u8::from(t) << 1),
                    (other, _, _) => other,
                })
                .collect();
            (shares_of(&codes, &sizes), codes)
        })
    }
}

impl NavMesh {
    /// The position itself when it lies over an area, otherwise the centroid of
    /// the nearest area within [`SNAP_DISTANCE`], otherwise `None`.
    fn on_mesh(&self, pos: [f32; 3]) -> Option<[f32; 3]> {
        if self.nav.find_area(pos).is_some() {
            return Some(pos);
        }
        let limit = SNAP_DISTANCE * SNAP_DISTANCE;
        self.centroids
            .iter()
            .map(|centroid| {
                let d = [
                    centroid[0] - pos[0],
                    centroid[1] - pos[1],
                    centroid[2] - pos[2],
                ];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], *centroid)
            })
            .filter(|(distance, _)| *distance <= limit)
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, centroid)| centroid)
    }

    fn travel(
        &self,
        from: [f32; 3],
        to: [f32; 3],
        speed: f32,
        by_distance: bool,
        params: &TerritoryParams,
    ) -> f32 {
        if by_distance {
            let (Some(start), Some(end)) = (self.on_mesh(from), self.on_mesh(to)) else {
                return f32::NAN;
            };
            let (Some(start), Some(end)) = (self.nav.find_area(start), self.nav.find_area(end))
            else {
                return f32::NAN;
            };
            let distances = self.nav.multi_source_distances(
                &[start],
                awpy::nav::PathWeight::Distance,
                &std::collections::HashSet::new(),
            );
            return distances.get(&end).map_or(f32::NAN, |d| *d as f32 / speed);
        }
        self.territory
            .travel_time(from, to, speed, &[], params)
            .unwrap_or(f32::NAN)
    }

    /// Height of the highest area under `(x, y)`.
    fn top_floor(&self, x: f32, y: f32) -> Option<f32> {
        self.nav
            .areas
            .iter()
            .filter(|area| area.contains_xy(x, y))
            .map(|area| area.centroid()[2])
            .max_by(f32::total_cmp)
    }
}
