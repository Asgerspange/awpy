//! Node.js bindings for awpy.
//!
//! One `Demo` class wraps an [`awpy::Parser`]. The parser memory-maps the demo
//! once and keeps its caches across calls. Per-tick outputs come back as
//! columns: typed arrays for numbers, and dictionary-encoded strings for
//! repeated text such as names, weapons and Steam ids. A full match of player
//! samples stays compact on the JavaScript side this way.
//!
//! Steam ids are returned as decimal strings. They do not fit in a JavaScript
//! number.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use awpy::{FieldValue, Parser, PlayerState};
use napi::bindgen_prelude::{Float32Array, Int32Array, Uint8Array, Uint32Array};
use napi_derive::napi;
use serde_json::Value;

fn js_error(error: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(error.to_string())
}

fn opt_f64(value: Option<f32>) -> Option<f64> {
    value.map(f64::from)
}

/// Demo header fields and playback info.
#[napi(object)]
pub struct Header {
    pub map_name: Option<String>,
    pub server_name: Option<String>,
    pub client_name: Option<String>,
    pub demo_version_name: Option<String>,
    pub build_num: Option<i32>,
    pub patch_version: Option<i32>,
    pub playback_ticks: Option<i32>,
    pub playback_time: Option<f64>,
    /// Ticks per second from playback info, or 64 when the demo lacks it.
    pub tick_rate: f64,
}

/// Strings stored once in `values` and referenced by index from `codes`.
/// A code of `-1` means null.
#[napi(object)]
pub struct StringColumn {
    pub values: Vec<String>,
    pub codes: Int32Array,
}

/// Builds a [`StringColumn`] row by row.
#[derive(Default)]
struct StringColumnBuilder {
    index: HashMap<String, i32>,
    steamids: HashMap<u64, i32>,
    values: Vec<String>,
    codes: Vec<i32>,
}

impl StringColumnBuilder {
    fn intern(&mut self, text: &str) -> i32 {
        if let Some(&code) = self.index.get(text) {
            return code;
        }
        let code = i32::try_from(self.values.len()).unwrap_or(i32::MAX);
        self.values.push(text.to_owned());
        self.index.insert(text.to_owned(), code);
        code
    }

    fn push(&mut self, value: Option<&str>) {
        let code = value.map_or(-1, |text| self.intern(text));
        self.codes.push(code);
    }

    /// Push a Steam id without formatting it again for every repeated row.
    fn push_steamid(&mut self, value: Option<u64>) {
        let code = match value {
            None => -1,
            Some(id) => match self.steamids.get(&id) {
                Some(&code) => code,
                None => {
                    let code = self.intern(&id.to_string());
                    self.steamids.insert(id, code);
                    code
                }
            },
        };
        self.codes.push(code);
    }

    /// Append another builder's rows, re-mapping its codes into this dictionary.
    fn append(&mut self, other: Self) {
        let remap: Vec<i32> = other.values.iter().map(|text| self.intern(text)).collect();
        self.codes.extend(other.codes.into_iter().map(|code| {
            usize::try_from(code)
                .ok()
                .and_then(|index| remap.get(index).copied())
                .unwrap_or(-1)
        }));
    }

    fn finish(self) -> StringColumn {
        StringColumn {
            values: self.values,
            codes: Int32Array::new(self.codes),
        }
    }
}

/// A participant of an [`Event`].
#[napi(object)]
pub struct EventPlayer {
    /// The event key this participant came from: `userid`, `attacker` or `assister`.
    pub role: String,
    pub steamid: Option<String>,
    pub name: Option<String>,
    pub side: Option<String>,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
    pub pitch: Option<f64>,
    pub yaw: Option<f64>,
}

/// A raw game event with its participants resolved at the event tick.
#[napi(object)]
pub struct Event {
    pub tick: i32,
    pub name: String,
    /// Raw event keys. Every value is a string, as decoded by awpy.
    pub keys: HashMap<String, String>,
    pub players: Vec<EventPlayer>,
}

/// Thrown-grenade trajectory samples, one row per tick a projectile moved.
#[napi(object)]
pub struct GrenadeColumns {
    pub tick: Int32Array,
    pub entity_id: Int32Array,
    pub entity_serial: Uint32Array,
    /// `smoke`, `he`, `flashbang`, `molotov`, `decoy` or `grenade`.
    pub kind: StringColumn,
    pub x: Float32Array,
    pub y: Float32Array,
    pub z: Float32Array,
    pub thrower_steamid: StringColumn,
}

/// A weapon purchase, pickup or drop.
#[napi(object)]
pub struct ItemEvent {
    pub tick: i32,
    pub action: String,
    pub steamid: Option<String>,
    pub name: Option<String>,
    pub side: Option<String>,
    pub item: String,
    pub cost: Option<i32>,
}

/// Player samples, one row per player per requested tick. Missing positions
/// are `NaN`. Flags are `0` or `1`.
#[napi(object)]
pub struct PlayerColumns {
    pub tick: Int32Array,
    pub steamid: StringColumn,
    pub name: StringColumn,
    /// `2` for T, `3` for CT, `0` otherwise.
    pub team_num: Uint8Array,
    pub x: Float32Array,
    pub y: Float32Array,
    pub z: Float32Array,
    pub pitch: Float32Array,
    pub yaw: Float32Array,
    pub health: Int32Array,
    pub armor: Int32Array,
    pub money: Int32Array,
    pub equipment_value: Int32Array,
    pub flash_duration: Float32Array,
    pub has_helmet: Uint8Array,
    pub has_defuser: Uint8Array,
    pub is_crouched: Uint8Array,
    pub is_walking: Uint8Array,
    pub is_airborne: Uint8Array,
    pub is_scoped: Uint8Array,
    pub is_defusing: Uint8Array,
    pub active_weapon: StringColumn,
    /// Comma-separated item names, in slot order.
    pub inventory: StringColumn,
}

/// Collects one parallel segment of player samples.
#[derive(Default)]
struct PlayerChunk {
    tick: Vec<i32>,
    team_num: Vec<u8>,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    pitch: Vec<f32>,
    yaw: Vec<f32>,
    health: Vec<i32>,
    armor: Vec<i32>,
    money: Vec<i32>,
    equipment_value: Vec<i32>,
    flash_duration: Vec<f32>,
    has_helmet: Vec<u8>,
    has_defuser: Vec<u8>,
    is_crouched: Vec<u8>,
    is_walking: Vec<u8>,
    is_airborne: Vec<u8>,
    is_scoped: Vec<u8>,
    is_defusing: Vec<u8>,
    steamid: StringColumnBuilder,
    name: StringColumnBuilder,
    active_weapon: StringColumnBuilder,
    inventory: StringColumnBuilder,
}

fn team_num(side: Option<&str>) -> u8 {
    match side {
        Some("terrorist") => 2,
        Some("counter-terrorist") => 3,
        _ => 0,
    }
}

impl PlayerChunk {
    fn push(&mut self, state: PlayerState) {
        self.tick.push(state.tick);
        self.team_num.push(team_num(state.side));
        self.x.push(state.x.unwrap_or(f32::NAN));
        self.y.push(state.y.unwrap_or(f32::NAN));
        self.z.push(state.z.unwrap_or(f32::NAN));
        self.pitch.push(state.pitch);
        self.yaw.push(state.yaw);
        self.health.push(state.health);
        self.armor.push(state.armor);
        self.money.push(state.money);
        self.equipment_value.push(state.equipment_value);
        self.flash_duration.push(state.flash_duration);
        self.has_helmet.push(u8::from(state.has_helmet));
        self.has_defuser.push(u8::from(state.has_defuser));
        self.is_crouched.push(u8::from(state.is_crouched));
        self.is_walking.push(u8::from(state.is_walking));
        self.is_airborne.push(u8::from(state.is_jumping));
        self.is_scoped.push(u8::from(state.is_scoped));
        self.is_defusing.push(u8::from(state.is_defusing));
        self.steamid.push_steamid(state.steamid);
        self.name.push(state.name.as_deref());
        self.active_weapon.push(state.active_weapon);
        self.inventory.push(Some(state.inventory.as_str()));
    }

    fn append(&mut self, other: Self) {
        self.tick.extend(other.tick);
        self.team_num.extend(other.team_num);
        self.x.extend(other.x);
        self.y.extend(other.y);
        self.z.extend(other.z);
        self.pitch.extend(other.pitch);
        self.yaw.extend(other.yaw);
        self.health.extend(other.health);
        self.armor.extend(other.armor);
        self.money.extend(other.money);
        self.equipment_value.extend(other.equipment_value);
        self.flash_duration.extend(other.flash_duration);
        self.has_helmet.extend(other.has_helmet);
        self.has_defuser.extend(other.has_defuser);
        self.is_crouched.extend(other.is_crouched);
        self.is_walking.extend(other.is_walking);
        self.is_airborne.extend(other.is_airborne);
        self.is_scoped.extend(other.is_scoped);
        self.is_defusing.extend(other.is_defusing);
        self.steamid.append(other.steamid);
        self.name.append(other.name);
        self.active_weapon.append(other.active_weapon);
        self.inventory.append(other.inventory);
    }

    fn finish(self) -> PlayerColumns {
        PlayerColumns {
            tick: Int32Array::new(self.tick),
            steamid: self.steamid.finish(),
            name: self.name.finish(),
            team_num: Uint8Array::new(self.team_num),
            x: Float32Array::new(self.x),
            y: Float32Array::new(self.y),
            z: Float32Array::new(self.z),
            pitch: Float32Array::new(self.pitch),
            yaw: Float32Array::new(self.yaw),
            health: Int32Array::new(self.health),
            armor: Int32Array::new(self.armor),
            money: Int32Array::new(self.money),
            equipment_value: Int32Array::new(self.equipment_value),
            flash_duration: Float32Array::new(self.flash_duration),
            has_helmet: Uint8Array::new(self.has_helmet),
            has_defuser: Uint8Array::new(self.has_defuser),
            is_crouched: Uint8Array::new(self.is_crouched),
            is_walking: Uint8Array::new(self.is_walking),
            is_airborne: Uint8Array::new(self.is_airborne),
            is_scoped: Uint8Array::new(self.is_scoped),
            is_defusing: Uint8Array::new(self.is_defusing),
            active_weapon: self.active_weapon.finish(),
            inventory: self.inventory.finish(),
        }
    }
}

impl Extend<PlayerState> for PlayerChunk {
    fn extend<I: IntoIterator<Item = PlayerState>>(&mut self, iter: I) {
        for state in iter {
            self.push(state);
        }
    }
}

/// Raw entity fields read at one tick. See [`Demo::entity_fields`].
#[napi(object)]
pub struct EntityFields {
    pub tick: i32,
    pub index: i32,
    /// Requested field paths to values. Missing fields are `null`. Unsigned
    /// values above 2^53 (Steam ids) are strings.
    pub values: HashMap<String, Value>,
}

/// Largest integer a JavaScript number holds exactly.
const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

fn field_json(value: &FieldValue) -> Value {
    let floats =
        |items: &[f32]| Value::from(items.iter().map(|f| f64::from(*f)).collect::<Vec<_>>());
    match value {
        FieldValue::Bool(v) => Value::from(*v),
        FieldValue::I32(v) => Value::from(*v),
        FieldValue::I64(v) => Value::from(*v),
        FieldValue::U32(v) => Value::from(*v),
        FieldValue::U64(v) if *v <= MAX_SAFE_INTEGER => Value::from(*v),
        FieldValue::U64(v) => Value::from(v.to_string()),
        FieldValue::F32(v) => Value::from(f64::from(*v)),
        FieldValue::String(bytes) => Value::from(String::from_utf8_lossy(bytes).into_owned()),
        FieldValue::Vector2(v) => floats(v),
        FieldValue::Vector3(v) | FieldValue::QAngle(v) => floats(v),
        FieldValue::Vector4(v) => floats(v),
        FieldValue::FloatVector(v) => floats(v),
        _ => Value::Null,
    }
}

/// A CS2 demo opened with awpy.
#[napi]
pub struct Demo {
    parser: Parser,
}

#[napi]
impl Demo {
    /// Open and verify a demo file.
    #[napi(constructor)]
    pub fn new(path: String) -> napi::Result<Self> {
        let parser = Parser::from_file(Path::new(&path)).map_err(js_error)?;
        parser.verify().map_err(js_error)?;
        Ok(Self { parser })
    }

    /// Header fields and the tick rate.
    #[napi]
    pub fn header(&self) -> napi::Result<Header> {
        let header = self.parser.file_header().map_err(js_error)?;
        let info = self.parser.file_info().ok();
        Ok(Header {
            map_name: header.map_name.clone(),
            server_name: header.server_name.clone(),
            client_name: header.client_name.clone(),
            demo_version_name: header.demo_version_name.clone(),
            build_num: header.build_num,
            patch_version: header.patch_version,
            playback_ticks: info.as_ref().and_then(|i| i.playback_ticks),
            playback_time: info.as_ref().and_then(|i| opt_f64(i.playback_time)),
            tick_rate: f64::from(self.parser.tickrate()),
        })
    }

    /// The named game events, with `userid`, `attacker` and `assister`
    /// resolved to players at the event tick.
    #[napi]
    pub fn events(&self, names: Vec<String>) -> napi::Result<Vec<Event>> {
        let wanted: HashSet<&str> = names.iter().map(String::as_str).collect();
        let rows = self.parser.events_with_players(&wanted).map_err(js_error)?;
        Ok(rows
            .into_iter()
            .map(|row| Event {
                tick: row.tick,
                name: row.name,
                keys: row.keys.into_iter().collect(),
                players: row
                    .players
                    .into_iter()
                    .map(|player| EventPlayer {
                        role: player.role.to_owned(),
                        steamid: player.steamid.map(|id| id.to_string()),
                        name: player.name,
                        side: player.side,
                        x: opt_f64(player.x),
                        y: opt_f64(player.y),
                        z: opt_f64(player.z),
                        pitch: opt_f64(player.pitch),
                        yaw: opt_f64(player.yaw),
                    })
                    .collect(),
            })
            .collect())
    }

    /// Thrown-grenade trajectories.
    #[napi]
    pub fn grenades(&self) -> napi::Result<GrenadeColumns> {
        let rows = self.parser.grenades().map_err(js_error)?;
        let mut tick = Vec::with_capacity(rows.len());
        let mut entity_id = Vec::with_capacity(rows.len());
        let mut entity_serial = Vec::with_capacity(rows.len());
        let mut x = Vec::with_capacity(rows.len());
        let mut y = Vec::with_capacity(rows.len());
        let mut z = Vec::with_capacity(rows.len());
        let mut kind = StringColumnBuilder::default();
        let mut thrower = StringColumnBuilder::default();
        for row in rows {
            tick.push(row.tick);
            entity_id.push(row.entity_id);
            entity_serial.push(row.entity_serial);
            x.push(row.x);
            y.push(row.y);
            z.push(row.z);
            kind.push(Some(row.grenade_type.as_str()));
            thrower.push_steamid(row.thrower_steamid);
        }
        Ok(GrenadeColumns {
            tick: Int32Array::new(tick),
            entity_id: Int32Array::new(entity_id),
            entity_serial: Uint32Array::new(entity_serial),
            kind: kind.finish(),
            x: Float32Array::new(x),
            y: Float32Array::new(y),
            z: Float32Array::new(z),
            thrower_steamid: thrower.finish(),
        })
    }

    /// Weapon purchases, pickups and drops.
    #[napi]
    pub fn item_events(&self) -> napi::Result<Vec<ItemEvent>> {
        let rows = self.parser.item_events().map_err(js_error)?;
        Ok(rows
            .into_iter()
            .map(|row| ItemEvent {
                tick: row.tick,
                action: row.action.to_string(),
                steamid: row.steamid.map(|id| id.to_string()),
                name: row.name,
                side: row.side.map(|side| side.to_string()),
                item: row.item.to_string(),
                cost: row.cost,
            })
            .collect())
    }

    /// Player samples at exactly the given ticks, decoded in parallel segments.
    /// Set `AWPY_TICK_SEGMENTS` to cap the number of threads.
    #[napi]
    pub fn players(&self, ticks: Int32Array) -> napi::Result<PlayerColumns> {
        let wanted: HashSet<i32> = ticks.iter().copied().collect();
        let (Some(&start), Some(&end)) = (wanted.iter().min(), wanted.iter().max()) else {
            return Ok(PlayerChunk::default().finish());
        };
        let chunks = self
            .parser
            .snapshots_query_chunks::<PlayerChunk>(None, &wanted, start, end)
            .map_err(js_error)?;
        let mut merged = PlayerChunk::default();
        for chunk in chunks {
            merged.append(chunk);
        }
        Ok(merged.finish())
    }

    /// Raw fields of every active entity of `class_name` at each given tick.
    /// Field names are serializer paths such as `m_iCompetitiveRanking` or
    /// `m_pInGameMoneyServices.m_iAccount`.
    #[napi]
    pub fn entity_fields(
        &self,
        ticks: Vec<i32>,
        class_name: String,
        fields: Vec<String>,
    ) -> napi::Result<Vec<EntityFields>> {
        let mut rows = Vec::new();
        for tick in ticks {
            let ctx = self.parser.parse_to_tick(tick).map_err(js_error)?;
            let Some(serializer) = ctx.serializers().get(&class_name) else {
                continue;
            };
            let keys: Vec<(&String, Option<u64>)> = fields
                .iter()
                .map(|field| (field, serializer.resolve_field_key(field)))
                .collect();
            for (_, entity) in ctx.entities().iter() {
                if !entity.active || entity.class_name.as_ref() != class_name {
                    continue;
                }
                let values = keys
                    .iter()
                    .map(|(field, key)| {
                        let value = key
                            .and_then(|key| entity.fields.get(&key))
                            .map_or(Value::Null, field_json);
                        ((*field).clone(), value)
                    })
                    .collect();
                rows.push(EntityFields {
                    tick: ctx.tick(),
                    index: entity.index,
                    values,
                });
            }
        }
        Ok(rows)
    }
}
