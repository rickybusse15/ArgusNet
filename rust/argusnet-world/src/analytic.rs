//! Analytic terrain evaluation. Preset parameters are data, not another sampler.
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Feature {
    #[serde(rename = "mountain-range-v1")]
    Mountain {
        start_xy: [f64; 2],
        end_xy: [f64; 2],
        peak_elevation_m: f64,
        width_m: f64,
        peak_count: i32,
        roughness: f64,
    },
    #[serde(rename = "valley-v1")]
    Valley {
        start_xy: [f64; 2],
        end_xy: [f64; 2],
        depth_m: f64,
        width_m: f64,
    },
    #[serde(rename = "ridgeline-v1")]
    Ridge {
        start_xy: [f64; 2],
        end_xy: [f64; 2],
        peak_elevation_m: f64,
        width_m: f64,
    },
    #[serde(rename = "plateau-v1")]
    Plateau {
        center_xy: [f64; 2],
        radius_m: f64,
        elevation_m: f64,
        edge_sharpness: f64,
    },
    #[serde(rename = "noise-layer-v1")]
    Noise {
        seed: i64,
        amplitude_m: f64,
        base_wavelength_m: f64,
        octaves: usize,
        persistence: f64,
        lacunarity: f64,
    },
}
fn segment_frame(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> (f64, f64) {
    let d = [b[0] - a[0], b[1] - a[1]];
    let len = d[0].hypot(d[1]).max(1e-9);
    let u = [d[0] / len, d[1] / len];
    let q = [p[0] - a[0], p[1] - a[1]];
    let along = (q[0] * u[0] + q[1] * u[1]).clamp(0.0, len);
    (
        along,
        (p[0] - a[0] - u[0] * along).hypot(p[1] - a[1] - u[1] * along),
    )
}
pub fn noise_hash(ix: i64, iy: i64, seed: i64) -> f64 {
    let mut h = ix.wrapping_mul(73856093) ^ iy.wrapping_mul(19349663) ^ seed;
    h = ((h >> 13) ^ h).wrapping_mul(0x45D9F3B) & 0xFFFFFFFF;
    h = ((h >> 13) ^ h).wrapping_mul(0x45D9F3B) & 0xFFFFFFFF;
    h = (h >> 13) ^ h;
    (h & 0xFFFF) as f64 / 32767.5 - 1.0
}
pub fn value_noise(x: f64, y: f64, seed: i64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let fx = smooth(x - x.floor());
    let fy = smooth(y - y.floor());
    let a = noise_hash(ix, iy, seed);
    let b = noise_hash(ix + 1, iy, seed);
    let c = noise_hash(ix, iy + 1, seed);
    let d = noise_hash(ix + 1, iy + 1, seed);
    let lo = a + fx * (b - a);
    lo + fy * (c + fx * (d - c) - lo)
}
impl Feature {
    pub fn height(&self, p: [f64; 2]) -> f64 {
        match *self {
            Self::Mountain {
                start_xy,
                end_xy,
                peak_elevation_m,
                width_m,
                peak_count,
                roughness,
            } => {
                let (along, cross) = segment_frame(p, start_xy, end_xy);
                let len = (end_xy[0] - start_xy[0])
                    .hypot(end_xy[1] - start_xy[1])
                    .max(width_m)
                    .max(1.0);
                let spacing = len / (peak_count + 1).max(1) as f64;
                (0..peak_count.max(1))
                    .map(|i| {
                        let amp = peak_elevation_m
                            * (0.8 + roughness * 0.2 * (((i + 1) as f64) * 1.618).sin());
                        amp * (-0.5
                            * ((along - spacing * (i + 1) as f64)
                                / (spacing * 0.9).max(width_m * 0.75).max(1.0))
                            .powi(2))
                        .exp()
                    })
                    .sum::<f64>()
                    * (-0.5 * (cross / width_m.max(1.0)).powi(2)).exp()
            }
            Self::Valley {
                start_xy,
                end_xy,
                depth_m,
                width_m,
            } => {
                -depth_m
                    * (-0.5 * (segment_frame(p, start_xy, end_xy).1 / width_m.max(1.0)).powi(2))
                        .exp()
            }
            Self::Ridge {
                start_xy,
                end_xy,
                peak_elevation_m,
                width_m,
            } => {
                peak_elevation_m
                    * (-0.5 * (segment_frame(p, start_xy, end_xy).1 / width_m.max(1.0)).powi(2))
                        .exp()
            }
            Self::Plateau {
                center_xy,
                radius_m,
                elevation_m,
                edge_sharpness,
            } => {
                elevation_m
                    / (1.0
                        + ((p[0] - center_xy[0]).hypot(p[1] - center_xy[1]) / radius_m.max(1.0))
                            .powf(edge_sharpness.max(1.0)))
            }
            Self::Noise {
                seed,
                amplitude_m,
                base_wavelength_m,
                octaves,
                persistence,
                lacunarity,
            } => {
                let mut wave = base_wavelength_m.max(1.0);
                let mut amp = 1.0;
                let mut total = 0.0;
                for i in 0..octaves {
                    total += amp * value_noise(p[0] / wave, p[1] / wave, seed + i as i64);
                    wave /= lacunarity;
                    amp *= persistence;
                }
                total * amplitude_m
            }
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticTerrain {
    pub ground_plane_m: f64,
    pub base_elevation_m: f64,
    pub slope_x_m_per_m: f64,
    pub slope_y_m_per_m: f64,
    pub wave_amplitude_m: f64,
    pub wave_length_x_m: f64,
    pub wave_length_y_m: f64,
    pub wave_phase_x_rad: f64,
    pub wave_phase_y_rad: f64,
    pub ridge_amplitude_m: f64,
    pub ridge_center_x_m: f64,
    pub ridge_center_y_m: f64,
    pub ridge_radius_m: f64,
    pub basin_depth_m: f64,
    pub basin_center_x_m: f64,
    pub basin_center_y_m: f64,
    pub basin_radius_m: f64,
    pub features: Vec<Feature>,
}
impl AnalyticTerrain {
    pub fn analytic_height(&self, p: [f64; 2]) -> f64 {
        let gaussian = |x: f64, y: f64, r: f64| {
            (-((p[0] - x).powi(2) + (p[1] - y).powi(2)) / (2.0 * (r * r).max(1.0))).exp()
        };
        self.base_elevation_m
            + p[0] * self.slope_x_m_per_m
            + p[1] * self.slope_y_m_per_m
            + self.wave_amplitude_m
                * (p[0] * TAU / self.wave_length_x_m.max(1.0) + self.wave_phase_x_rad).sin()
                * (p[1] * TAU / self.wave_length_y_m.max(1.0) + self.wave_phase_y_rad).cos()
            + self.ridge_amplitude_m
                * gaussian(
                    self.ridge_center_x_m,
                    self.ridge_center_y_m,
                    self.ridge_radius_m,
                )
            - self.basin_depth_m
                * gaussian(
                    self.basin_center_x_m,
                    self.basin_center_y_m,
                    self.basin_radius_m,
                )
            + self.features.iter().map(|f| f.height(p)).sum::<f64>()
    }
    pub fn height(&self, p: [f64; 2]) -> f64 {
        self.analytic_height(p).max(self.ground_plane_m)
    }
}
