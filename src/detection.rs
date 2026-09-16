use anyhow::{Result, ensure};
use serde::Serialize;
/// Packed RGB pixels, three bytes per pixel with no row padding.
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub rgb: &'a [u8],
    pub width: usize,
    pub height: usize,
}
impl Image<'_> {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.width > 0
                && self.height > 0
                && self
                    .width
                    .checked_mul(self.height)
                    .and_then(|n| n.checked_mul(3))
                    == Some(self.rgb.len()),
            "invalid RGB image size"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub landmarks: [[f32; 2]; 5],
}
#[derive(Clone, Copy, Debug, Default)]
pub enum Ranking {
    #[default]
    AreaAndCenter,
    Area,
}
#[derive(Clone, Copy, Debug)]
pub struct DetectionOptions {
    pub threshold: f32,
    pub nms_threshold: f32,
    pub max_candidates: usize,
    pub max_detections: usize,
    pub ranking: Ranking,
}
impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            nms_threshold: 0.4,
            max_candidates: 4096,
            max_detections: 0,
            ranking: Ranking::AreaAndCenter,
        }
    }
}
impl DetectionOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.threshold > 0.
                && self.threshold < 1.
                && self.nms_threshold > 0.
                && self.nms_threshold < 1.
                && self.max_candidates > 0,
            "invalid detection options"
        );
        Ok(())
    }
}
/// Aspect-preserving resize with black padding at the right and bottom.
#[cfg(test)]
pub(crate) fn letterbox(image: Image<'_>) -> Result<(Vec<u8>, f32)> {
    let mut out = vec![0; 640 * 640 * 3];
    let scale = letterbox_into(image, &mut out, &mut fast_image_resize::Resizer::new())?;
    Ok((out, scale))
}
#[cfg(test)]
fn letterbox_into(
    image: Image<'_>,
    out: &mut [u8],
    resizer: &mut fast_image_resize::Resizer,
) -> Result<f32> {
    image.validate()?;
    ensure!(out.len() == 640 * 640 * 3, "invalid canvas size");
    let (w, h) = if image.height > image.width {
        (640 * image.width / image.height, 640)
    } else {
        (640, 640 * image.height / image.width)
    };
    ensure!(w > 0 && h > 0, "image aspect ratio is too extreme");
    out.fill(0);
    use fast_image_resize::{
        FilterType, PixelType, ResizeAlg, ResizeOptions,
        images::{CroppedImageMut, Image as ResizeImage, ImageRef},
    };
    let source = ImageRef::new(
        u32::try_from(image.width)?,
        u32::try_from(image.height)?,
        image.rgb,
        PixelType::U8x3,
    )?;
    let mut target = ResizeImage::from_slice_u8(640, 640, out, PixelType::U8x3)?;
    let mut cropped = CroppedImageMut::new(&mut target, 0, 0, w as u32, h as u32)?;
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear));
    resizer.resize(&source, &mut cropped, &options)?;
    Ok(h as f32 / image.height as f32)
}
#[cfg(test)]
pub(crate) fn decode(
    heads: &[Vec<half::f16>],
    batch: usize,
    scale: f32,
    shape: [usize; 2],
    options: DetectionOptions,
) -> Result<Vec<Detection>> {
    let mut candidates = vec![];
    // Relax the logit bound before the exact sigmoid test so rounding at the
    // probability threshold cannot drop a candidate. Most anchors need no exp.
    let bound = (options.threshold / (1. - options.threshold)).ln() - 1e-3;
    for (level, size) in [80, 40, 20].into_iter().enumerate() {
        let stride = (640 / size) as f32;
        for y in 0..size {
            for x in 0..size {
                let row = &heads[level][((batch * size + y) * size + x) * 64..][..64];
                for a in 0..2 {
                    let logit = row[a].to_f32();
                    ensure!(logit.is_finite(), "non-finite detector logit");
                    if logit < bound {
                        continue;
                    }
                    let score = 1. / (1. + (-logit).exp());
                    if score < options.threshold {
                        continue;
                    }
                    ensure!(
                        candidates.len() < options.max_candidates,
                        "candidate capacity exceeded"
                    );
                    let cx = x as f32 * stride;
                    let cy = y as f32 * stride;
                    let b = &row[2 + 4 * a..];
                    let k = &row[10 + 10 * a..];
                    let bbox = [
                        (cx - b[0].to_f32() * stride) / scale,
                        (cy - b[1].to_f32() * stride) / scale,
                        (cx + b[2].to_f32() * stride) / scale,
                        (cy + b[3].to_f32() * stride) / scale,
                    ];
                    let landmarks = std::array::from_fn(|i| {
                        [
                            (cx + k[2 * i].to_f32() * stride) / scale,
                            (cy + k[2 * i + 1].to_f32() * stride) / scale,
                        ]
                    });
                    ensure!(
                        score.is_finite()
                            && bbox
                                .iter()
                                .chain(landmarks.iter().flatten())
                                .all(|x| x.is_finite()),
                        "non-finite detector output"
                    );
                    candidates.push(Detection {
                        bbox,
                        score,
                        landmarks,
                    });
                }
            }
        }
    }
    Ok(suppress_reference(candidates, shape, options))
}
/// Inclusive-coordinate greedy NMS, followed by optional InsightFace ranking.
pub fn suppress(
    candidates: Vec<Detection>,
    shape: [usize; 2],
    options: DetectionOptions,
) -> Vec<Detection> {
    let compact: Vec<_> = candidates
        .iter()
        .map(|d| [d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]])
        .collect();
    select_candidates(&compact, shape, options)
        .into_iter()
        .map(|i| candidates[i].clone())
        .collect()
}

/// Select original row indices using only scores and boxes. Coordinates needed
/// by downstream models need not cross the device/host boundary.
pub(crate) fn select_candidates(
    candidates: &[[f32; 5]],
    shape: [usize; 2],
    options: DetectionOptions,
) -> Vec<usize> {
    let mut order: Vec<_> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| candidates[b][0].total_cmp(&candidates[a][0]));
    let bbox = |i: usize| <&[f32; 4]>::try_from(&candidates[i][1..]).unwrap();
    let mut kept = Vec::new();
    for i in order {
        if kept
            .iter()
            .all(|&j| iou(bbox(i), bbox(j)) <= options.nms_threshold)
        {
            kept.push(i);
        }
    }
    if options.max_detections > 0 && kept.len() > options.max_detections {
        let rank = |i: usize| {
            let b = bbox(i);
            let area = (b[2] - b[0]) * (b[3] - b[1]);
            match options.ranking {
                Ranking::Area => area,
                Ranking::AreaAndCenter => {
                    area - 2.
                        * (((b[0] + b[2]) / 2. - (shape[0] / 2) as f32).powi(2)
                            + ((b[1] + b[3]) / 2. - (shape[1] / 2) as f32).powi(2))
                }
            }
        };
        kept.sort_by(|&a, &b| rank(b).total_cmp(&rank(a)));
        kept.truncate(options.max_detections);
    }
    kept
}

// Frozen pre-migration host oracle, independent of the indexed selector.
#[cfg(test)]
fn suppress_reference(
    mut candidates: Vec<Detection>,
    shape: [usize; 2],
    options: DetectionOptions,
) -> Vec<Detection> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = vec![];
    for d in candidates {
        if kept
            .iter()
            .all(|k| iou(&d.bbox, &k.bbox) <= options.nms_threshold)
        {
            kept.push(d);
        }
    }
    if options.max_detections > 0 && kept.len() > options.max_detections {
        let rank = |d: &Detection| {
            let b = d.bbox;
            let area = (b[2] - b[0]) * (b[3] - b[1]);
            match options.ranking {
                Ranking::Area => area,
                Ranking::AreaAndCenter => {
                    area - 2.
                        * (((b[0] + b[2]) / 2. - (shape[0] / 2) as f32).powi(2)
                            + ((b[1] + b[3]) / 2. - (shape[1] / 2) as f32).powi(2))
                }
            }
        };
        kept.sort_by(|a, b| rank(b).total_cmp(&rank(a)));
        kept.truncate(options.max_detections);
    }
    kept
}
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter = (a[2].min(b[2]) - a[0].max(b[0]) + 1.).max(0.)
        * (a[3].min(b[3]) - a[1].max(b[1]) + 1.).max(0.);
    inter
        / ((a[2] - a[0] + 1.) * (a[3] - a[1] + 1.) + (b[2] - b[0] + 1.) * (b[3] - b[1] + 1.)
            - inter)
}
