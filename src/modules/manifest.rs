//! Parsing for the `modules.json` manifest at the root of a
//! sentence-transformers model directory.
//!
//! Example manifest (from `sentence-transformers/all-MiniLM-L6-v2`):
//! ```json
//! [
//!   { "idx": 0, "name": "0", "path": "",          "type": "sentence_transformers.models.Transformer" },
//!   { "idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling" },
//!   { "idx": 2, "name": "2", "path": "2_Normalize","type": "sentence_transformers.models.Normalize" }
//! ]
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::Error;

/// A single entry in `modules.json`. The `path` field is the subdirectory
/// (relative to the model root) that holds this module's config/weights.
#[derive(Debug, Deserialize)]
pub struct ModuleEntry {
    pub idx: u32,
    pub name: String,
    pub path: String,
    /// Fully-qualified Python class path, e.g.
    /// `sentence_transformers.models.Transformer`. We dispatch on the part
    /// after the last `.`.
    #[serde(rename = "type")]
    pub module_type: String,
}

impl ModuleEntry {
    /// Last segment of `module_type` — `"Transformer"`, `"Pooling"`,
    /// `"Dense"`, `"Normalize"`, etc. Lowercased for case-insensitive matching.
    pub fn short_type(&self) -> &str {
        self.module_type
            .rsplit('.')
            .next()
            .unwrap_or(self.module_type.as_str())
    }
}

/// Read and parse `<root>/modules.json`. Errors out if the file is missing
/// or malformed — every sentence-transformers checkpoint must ship one.
pub fn parse(root: &Path) -> Result<Vec<ModuleEntry>, Error> {
    let modules_file = root.join("modules.json");
    let content = std::fs::read_to_string(&modules_file).map_err(|e| {
        log::error!(
            "modules.json not readable at {}: {}",
            modules_file.display(),
            e
        );
        Error::Encoding("missing or unreadable modules.json (not a sentence-transformers checkpoint?)")
    })?;
    let entries: Vec<ModuleEntry> = serde_json::from_str(&content).map_err(|e| {
        log::error!("invalid modules.json: {}", e);
        Error::Encoding("invalid modules.json")
    })?;
    if entries.is_empty() {
        return Err(Error::Encoding("modules.json is empty"));
    }
    Ok(entries)
}

/// Resolve a module entry to its on-disk directory.
///
/// The `path` field in real-world `modules.json` files is unreliable:
///
/// * Modern HF checkpoints (e.g. `all-MiniLM-L6-v2`) declare `"path": ""`
///   for the Transformer — files live at the model root, not in a subdir.
/// * Some legacy checkpoints (e.g. the original UKPabs
///   `distiluse-base-multilingual-cased` export) declare
///   `"path": "0_Transformer"` but the actual on-disk directory is named
///   `0_DistilBERT`. The original rust-sbert worked around this by
///   hard-coding `0_DistilBERT` and ignoring `modules.json` entirely.
///
/// To stay faithful to `modules.json` while tolerating both cases, this
/// resolver applies the following algorithm:
///
/// 1. If `path` is empty → return `root` (files at model root).
/// 2. If `<root>/<path>` exists → return it.
/// 3. Otherwise scan `<root>` for a directory named `<idx>_*` and return
///    the first match. This covers the distiluse case (manifest says
///    `0_Transformer`, on-disk is `0_DistilBERT`, both start with `0_`).
/// 4. Fall back to returning the declared path anyway (so the downstream
///    file-open produces the usual "not found" error with the right path).
pub fn resolve_module_dir(root: &Path, entry: &ModuleEntry) -> PathBuf {
    // Case 1: empty path means files live at the model root.
    if entry.path.is_empty() {
        return root.to_path_buf();
    }

    let declared = root.join(&entry.path);
    if declared.exists() {
        return declared;
    }

    // Case 3: scan for `<idx>_*` sibling directory. The `idx` field is
    // reliable even when `path` lies.
    let prefix = format!("{}_", entry.idx);
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            if let Some(name) = e.file_name().to_str() {
                if name.starts_with(&prefix) && e.path().is_dir() {
                    log::warn!(
                        "modules.json entry {:?} declares path \"{}\" which does not exist; \
                         falling back to discovered directory \"{}\"",
                        entry.name,
                        entry.path,
                        name
                    );
                    return e.path();
                }
            }
        }
    }

    // Case 4: nothing better to return; let the caller's file-open fail
    // with a message that references the declared (expected) path.
    declared
}
