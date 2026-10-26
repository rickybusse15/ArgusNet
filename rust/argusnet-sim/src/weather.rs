//! The two weather reads motion makes: wind and the flight-speed penalty.
//!
//! `trajectory.rs` defined [`WeatherSampler`] as the capability and left the
//! model for later; this is that model, for exactly the fields motion reads.
//! `WindModel` is ported whole (it is one method). Of `WeatherModel`, the
//! motion reads (`flight_speed_penalty`, via wind and precipitation drag) came
//! first (P4.8e.1); the sensing reads -- `visibility_at_range`,
//! `bearing_noise_scale` and stage 6's cloud factor -- followed in P4.8e.2,
//! when stage 6 started using the world's detection model.
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

    /// `visibility_reduction_factor`: multiplicative, in `(0, 1]`.
    pub fn visibility_reduction_factor(&self) -> f64 {
        let rate = self.rate_mmph.max(0.0);
        match self.precip_type.as_str() {
            "none" => 1.0,
            "fog" => 0.15,
            _ if rate <= 0.0 => 1.0,
            "rain" => (1800.0 / rate.powf(0.6).max(0.01) / 10000.0).min(1.0),
            "snow" => (1200.0 / rate.powf(0.75).max(0.01) / 10000.0).min(1.0),
            "sleet" => (1500.0 / rate.powf(0.65).max(0.01) / 10000.0).min(1.0),
            _ => 1.0,
        }
    }

    /// `sensor_noise_multiplier`: always `>= 1`.
    pub fn sensor_noise_multiplier(&self) -> f64 {
        let rate = self.rate_mmph.max(0.0);
        match self.precip_type.as_str() {
            "none" => 1.0,
            "fog" => 1.6,
            _ if rate <= 0.0 => 1.0,
            "rain" => 1.0 + 0.15 * rate.powf(0.5),
            "snow" => 1.0 + 0.25 * rate.powf(0.5),
            "sleet" => 1.0 + 0.20 * rate.powf(0.5),
            _ => 1.0,
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

/// The two `AtmosphericConditions` fields a weather read uses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Atmosphere {
    pub visibility_m: f64,
    pub relative_humidity: f64,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            visibility_m: 10_000.0,
            relative_humidity: 0.5,
        }
    }
}

/// `CloudLayer`: a slab between two altitudes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CloudLayer {
    pub base_altitude_m: f64,
    pub top_altitude_m: f64,
    pub coverage: f64,
}

impl CloudLayer {
    /// A hard block only at full coverage, with the slab strictly between.
    pub fn obscures_los(&self, observer_alt_m: f64, target_alt_m: f64) -> bool {
        if self.coverage < 1.0 {
            return false;
        }
        let low = observer_alt_m.min(target_alt_m);
        let high = observer_alt_m.max(target_alt_m);
        low < self.base_altitude_m && high > self.top_altitude_m
    }
}

/// `WeatherModel`, as the scenario bundle carries it.
///
/// Named for its first consumer, motion; it now also carries what sensing
/// reads. Kept under this name because the bundle format and three call sites
/// use it, and the name is no longer the whole truth.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MotionWeather {
    pub wind: WindModel,
    pub precipitation: Precipitation,
    pub atmosphere: Atmosphere,
    pub cloud_layers: Vec<CloudLayer>,
}

impl MotionWeather {
    /// The meteorological visibility after precipitation, floored at 1 m:
    /// the `V` of [`Self::visibility_at_range`]'s Koschmieder law.
    pub fn effective_visibility_m(&self) -> f64 {
        (self.atmosphere.visibility_m * self.precipitation.visibility_reduction_factor()).max(1.0)
    }

    /// `visibility_at_range`: Koschmieder over the precipitation-reduced
    /// visibility, floored at 1 m.
    pub fn visibility_at_range(&self, range_m: f64) -> f64 {
        let precip_factor = self.precipitation.visibility_reduction_factor();
        let effective_vis = (self.atmosphere.visibility_m * precip_factor).max(1.0);
        let sigma = 3.912 / effective_vis;
        (-sigma * range_m.max(0.0)).exp().max(0.0)
    }

    /// `bearing_noise_scale`: precipitation noise times a humidity penalty.
    pub fn bearing_noise_scale(&self) -> f64 {
        let precip_noise = self.precipitation.sensor_noise_multiplier();
        let humidity_noise = 1.0 + 0.3 * (self.atmosphere.relative_humidity - 0.8).max(0.0);
        (precip_noise * humidity_noise).max(1.0)
    }

    /// Stage 6's cloud factor for one sight line: 0 through a full slab,
    /// otherwise the product of partial-coverage penalties. Not a
    /// `WeatherModel` method in the reference -- it is inline in
    /// `build_observations` -- and ported from there.
    pub fn cloud_factor(&self, observer_alt_m: f64, target_alt_m: f64) -> f64 {
        let mut factor = 1.0;
        for cloud in &self.cloud_layers {
            if cloud.obscures_los(observer_alt_m, target_alt_m) {
                return 0.0;
            }
            let low = observer_alt_m.min(target_alt_m);
            let high = observer_alt_m.max(target_alt_m);
            if low < cloud.top_altitude_m && high > cloud.base_altitude_m {
                factor *= (1.0 - cloud.coverage * 0.7).max(0.05);
            }
        }
        factor
    }
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
