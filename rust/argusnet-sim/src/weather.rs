//! The two weather reads motion makes: wind and the flight-speed penalty.
//!
//! `trajectory.rs` defined [`WeatherSampler`] as the capability and left the
//! model for later; this is that model, for exactly the fields motion reads.
//! `WindModel` is ported whole (it is one method). Of `WeatherModel`, only
//! `flight_speed_penalty` is here, which needs the wind and the precipitation
//! *drag* term. Visibility, bearing-noise scaling and cloud occlusion are
//! sensing's reads and belong with stage 6's weather wiring.
//!
//! Serializable because it arrives in the scenario bundle as parameters, not
//! as a preset name: the bundle is resolved data, so a preset table edited in
//! Python cannot silently change a bundle already written.

use serde::{Deserialize, Serialize};

use crate::trajectory::WeatherSampler;

/// `WindModel`: a steady base plus a sinusoidal gust, scaled with altitude.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindModel {
    pub base_speed_mps: f64,
    /// Direction the wind blows *from*, clockwise from north.
    pub base_heading_rad: f64,
    pub gust_amplitude_mps: f64,
    /// `<= 0` disables gusts.
    pub gust_period_s: f64,
    /// Fractional speed increase per 100 m of altitude.
    pub altitude_scaling: f64,
}

impl Default for WindModel {
    fn default() -> Self {
        Self {
            base_speed_mps: 0.0,
            base_heading_rad: 0.0,
            gust_amplitude_mps: 0.0,
            gust_period_s: 10.0,
            altitude_scaling: 0.0,
        }
    }
}

impl WindModel {
    /// `[east, north, 0]`, pointing downwind.
    ///
    /// The docstring says `[east, north, down]`; the third component is always
    /// zero in this model, so the label cannot matter yet. Carried, not
    /// renamed, so a vertical wind added later lands in the slot the
    /// reference documents.
    pub fn wind_at(&self, altitude_m: f64, time_s: f64) -> [f64; 3] {
        let alt_factor = 1.0 + self.altitude_scaling * altitude_m.max(0.0) / 100.0;
        let gust = if self.gust_period_s > 0.0 && self.gust_amplitude_mps > 0.0 {
            // `2.0 * math.pi * time_s / period`, evaluated left to right.
            self.gust_amplitude_mps
                * (2.0 * std::f64::consts::PI * time_s / self.gust_period_s).sin()
        } else {
            0.0
        };
        let speed = ((self.base_speed_mps + gust) * alt_factor).max(0.0);
        [
            -speed * self.base_heading_rad.sin(),
            -speed * self.base_heading_rad.cos(),
            0.0,
        ]
    }
}

/// `PrecipitationModel`'s type and rate; only the drag term is read here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Precipitation {
    /// `none`, `rain`, `snow`, `sleet` or `fog`.
    pub precip_type: String,
    pub rate_mmph: f64,
}

impl Default for Precipitation {
    fn default() -> Self {
        Self {
            precip_type: "none".to_string(),
            rate_mmph: 0.0,
        }
    }
}

impl Precipitation {
    /// The reference's validation, as a refusal rather than a panic.
    pub fn validate(&self) -> Result<(), String> {
        match self.precip_type.as_str() {
            "none" | "rain" | "snow" | "sleet" | "fog" => Ok(()),
            other => Err(format!("unknown precipitation type {other:?}")),
        }
    }

    fn drag(&self) -> f64 {
        match self.precip_type.as_str() {
            "rain" => (1.0 - 0.02 * self.rate_mmph).max(0.5),
            "snow" => (1.0 - 0.04 * self.rate_mmph).max(0.4),
            "sleet" => (1.0 - 0.03 * self.rate_mmph).max(0.45),
            "fog" => 0.9,
            _ => 1.0,
        }
    }
}

/// The motion half of `WeatherModel`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MotionWeather {
    pub wind: WindModel,
    pub precipitation: Precipitation,
}

impl WeatherSampler for MotionWeather {
    /// `flight_speed_penalty`. Wind is evaluated at **time zero** -- the
    /// reference's, and the reason [`WeatherSampler`] gives this read no time.
    fn flight_speed_penalty(&self, altitude_m: f64) -> f64 {
        const NOMINAL_CRUISE_MPS: f64 = 15.0;
        let wind = self.wind.wind_at(altitude_m, 0.0);
        // `np.linalg.norm` of a 3-vector: sqrt of the sum of squares, not
        // `hypot`. See `follow_path::norm`.
        let wind_speed = (wind[0] * wind[0] + wind[1] * wind[1] + wind[2] * wind[2]).sqrt();
        let wind_penalty = (1.0 - 0.5 * (wind_speed / NOMINAL_CRUISE_MPS)).max(0.1);
        (wind_penalty * self.precipitation.drag()).max(0.05)
    }

    fn wind_at(&self, altitude_m: f64, time_s: f64) -> [f64; 3] {
        self.wind.wind_at(altitude_m, time_s)
    }
}
