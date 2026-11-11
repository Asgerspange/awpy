//! Map control for Node.js: one navigation mesh, many snapshots per call.
//!
//! The caller already holds player positions (it sampled them while parsing),
//! so this takes positions instead of re-reading the demo. Each sample is
//! labelled with [`reachability_control`]; the samples are split across
//! threads, capped by `AWPY_TICK_SEGMENTS` like the demo decoder.

use std::num::NonZeroUsize;
use std::path::Path;

use awpy::map_control::{Observer, Occluder, Params, Team, reachability_control};
use awpy::nav::Nav;
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
}

/// One sample's inputs, borrowed out of [`ControlInput`].
struct Sample {
    observers: Vec<Observer>,
    fires: Vec<Occluder>,
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

#[napi]
impl NavMesh {
    /// Load a `.nav` file.
    #[napi(constructor)]
    pub fn new(path: String) -> napi::Result<Self> {
        let nav = Nav::from_file(Path::new(&path)).map_err(js_error)?;
        let centroids = nav.areas.iter().map(awpy::nav::NavArea::centroid).collect();
        Ok(Self { nav, centroids })
    }

    #[napi(getter)]
    pub fn area_count(&self) -> u32 {
        u32::try_from(self.nav.area_count()).unwrap_or(u32::MAX)
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

    /// Reachability map control for each sample: an area belongs to the side
    /// that can walk to it first, a near-tie is contested, and fires block the
    /// ground under them.
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
                        let team = match input.team[row] {
                            3 => Team::Ct,
                            2 => Team::T,
                            _ => return None,
                        };
                        let pos = [input.x[row], input.y[row], input.z[row]];
                        if pos.iter().any(|value| !value.is_finite()) {
                            return None;
                        }
                        Some(Observer {
                            pos: self.on_mesh(pos)?,
                            team,
                            crouched: false,
                            blind: false,
                            yaw: None,
                        })
                    })
                    .collect(),
                fires: (fires[0] as usize..fires[1] as usize)
                    .map(|row| Occluder {
                        center: [input.fire_x[row], input.fire_y[row], input.fire_z[row]],
                        radius,
                    })
                    .collect(),
            })
            .collect();

        let params = Params {
            contest_margin: input
                .contest_margin
                .unwrap_or(Params::default().contest_margin),
            ..Params::default()
        };
        let row_bytes = self.nav.area_count().div_ceil(4);
        let threads = thread_budget().min(samples.len().div_ceil(8)).max(1);
        let per_thread = samples.len().div_ceil(threads).max(1);
        let parts: Vec<(Vec<f32>, Vec<u8>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = samples
                .chunks(per_thread)
                .map(|chunk| {
                    let params = &params;
                    scope.spawn(move || self.label_chunk(chunk, params, row_bytes))
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

    fn label_chunk(
        &self,
        samples: &[Sample],
        params: &Params,
        row_bytes: usize,
    ) -> (Vec<f32>, Vec<u8>) {
        let mut shares = Vec::with_capacity(samples.len() * 4);
        let mut labels = vec![0u8; samples.len() * row_bytes];
        for (index, sample) in samples.iter().enumerate() {
            let control = reachability_control(&self.nav, &sample.observers, &sample.fires, params);
            shares.extend_from_slice(&[
                control.ct_fraction,
                control.t_fraction,
                control.contested_fraction,
                control.neutral_fraction,
            ]);
            let row = &mut labels[index * row_bytes..(index + 1) * row_bytes];
            for (area, label) in control.areas.iter().enumerate() {
                let code = u8::from(label.ct) | (u8::from(label.t) << 1);
                row[area / 4] |= code << ((area % 4) * 2);
            }
        }
        (shares, labels)
    }
}
