//! Pinned model weights in the shared Hugging Face cache.

use anyhow::Result;
use hrx::artifacts::hf::{HubFile, Repository, Resolver};
use std::path::PathBuf;

/// Hugging Face model repository.
pub const REPO: (&str, &str) = ("immich-app", "buffalo_l");
/// Revision whose weights are covered by this crate's numerical tests.
pub const REVISION: &str = "d09715916a0778919a770c343533641e250b8699";
/// Original model file within the repository.
pub const FILE: &str = "detection/model.onnx";

/// Resolve the pinned weights, downloading only on a cache miss.
pub fn weights(offline: bool) -> Result<PathBuf> {
    Ok(Resolver::new(Repository::new(REPO.0, REPO.1).at(REVISION))
        .offline(offline)
        .resolve(&HubFile::new(FILE))?)
}
