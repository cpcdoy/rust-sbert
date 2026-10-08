//! Tokenizer settings resolved from a checkpoint's config files — shared by
//! the torch and onnx drivers.

use std::path::Path;

/// Tokenizer settings resolved from a checkpoint's config files.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TokenizerSettings {
    pub do_lower_case: bool,
    pub max_seq_length: usize,
}

/// Resolve `do_lower_case` and `max_seq_length` for a checkpoint laid out as
/// `transformer_dir` (the module dir of the Transformer entry) plus the model
/// `root`.
///
/// Resolution order — first match wins:
///
/// * `do_lower_case`:
///   1. `tokenizer_config.json` (transformer dir, then model root) — the
///      authoritative HF tokenizer flag,
///   2. `sentence_bert_config.json` (model root, then transformer dir),
///   3. legacy `sentence_distilbert_config.json` (v0.2-era exports, in the
///      transformer subdir),
///   4. `false` (conservative: cased vocabs like distiluse must not be
///      folded).
///
///   `tokenizer_config.json` must take precedence: e.g. all-MiniLM-L6-v2 has
///   `do_lower_case: true` there but `do_lower_case: false` in
///   `sentence_bert_config.json`. Its vocab is lowercase-only, so the `true`
///   is required — tokenizing cased text case-sensitively maps every
///   capitalized word to `[UNK]`, collapsing distinct inputs
///   ("Tech. root" vs "Browsers. root") to byte-identical embeddings.
///
/// * `max_seq_length`:
///   1. `max_seq_length` of `sentence_bert_config.json` / legacy
///      `sentence_distilbert_config.json` — e.g. 256 for all-MiniLM-L6-v2;
///      truncating at a smaller length diverges from Python
///      sentence-transformers on long inputs,
///   2. `128` — the historical default of this crate (also what the v0.2
///      distiluse export declares).
pub(crate) fn resolve_tokenizer_settings(transformer_dir: &Path, root: &Path) -> TokenizerSettings {
    let read_json = |dir: &Path, file: &str| -> Option<serde_json::Value> {
        let path = dir.join(file);
        let value = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        if value.is_some() {
            log::info!("reading tokenizer settings from {}", path.display());
        }
        value
    };

    let tokenizer_config = [&transformer_dir, &root]
        .iter()
        .find_map(|dir| read_json(dir, "tokenizer_config.json"));

    let sbert_config = [&root, &transformer_dir]
        .iter()
        .find_map(|dir| read_json(dir, "sentence_bert_config.json"))
        .or_else(|| read_json(transformer_dir, "sentence_distilbert_config.json"));

    let do_lower_case = tokenizer_config
        .as_ref()
        .and_then(|v| v.get("do_lower_case").and_then(|b| b.as_bool()))
        .or_else(|| {
            sbert_config
                .as_ref()
                .and_then(|v| v.get("do_lower_case").and_then(|b| b.as_bool()))
        })
        .unwrap_or(false);

    let max_seq_length = sbert_config
        .as_ref()
        .and_then(|v| v.get("max_seq_length").and_then(|n| n.as_u64()))
        .map(|n| n as usize)
        .unwrap_or(128);

    TokenizerSettings {
        do_lower_case,
        max_seq_length,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmpdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sbert-settings-test-{}", name));
        fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, file: &str, json: &str) {
        fs::write(dir.join(file), json).unwrap();
    }

    #[test]
    fn defaults_when_no_config_present() {
        let root = tmpdir("defaults-root");
        let tdir = root.join("0_BERT");
        fs::create_dir_all(&tdir).unwrap();
        assert_eq!(
            resolve_tokenizer_settings(&tdir, &root),
            TokenizerSettings {
                do_lower_case: false,
                max_seq_length: 128,
            }
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn tokenizer_config_takes_precedence_over_sentence_bert_config() {
        // The all-MiniLM-L6-v2 situation: tokenizer_config.json says
        // do_lower_case=true, sentence_bert_config.json says false (and
        // carries max_seq_length=256). The HF tokenizer flag must win.
        let root = tmpdir("minilm-root");
        write(&root, "tokenizer_config.json", r#"{"do_lower_case": true}"#);
        write(
            &root,
            "sentence_bert_config.json",
            r#"{"max_seq_length": 256, "do_lower_case": false}"#,
        );
        // Modern layout: `path: ""` → transformer dir IS the model root.
        let s = resolve_tokenizer_settings(&root, &root);
        assert!(s.do_lower_case);
        assert_eq!(s.max_seq_length, 256);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn sentence_bert_config_used_when_no_tokenizer_config() {
        let root = tmpdir("sbertcfg-root");
        write(
            &root,
            "sentence_bert_config.json",
            r#"{"max_seq_length": 256, "do_lower_case": false}"#,
        );
        let s = resolve_tokenizer_settings(&root, &root);
        assert!(!s.do_lower_case);
        assert_eq!(s.max_seq_length, 256);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn legacy_v02_layout_reads_transformer_subdir_configs() {
        // UKP v0.2 distiluse: no root tokenizer_config.json; the transformer
        // subdir carries tokenizer_config.json + the legacy
        // sentence_distilbert_config.json.
        let root = tmpdir("v02-root");
        let tdir = root.join("0_DistilBERT");
        fs::create_dir_all(&tdir).unwrap();
        write(
            &tdir,
            "tokenizer_config.json",
            r#"{"do_lower_case": false, "max_len": 512}"#,
        );
        write(
            &tdir,
            "sentence_distilbert_config.json",
            r#"{"max_seq_length": 96, "do_lower_case": false}"#,
        );
        let s = resolve_tokenizer_settings(&tdir, &root);
        assert!(!s.do_lower_case);
        // 96 is deliberately non-default: 128 would be indistinguishable
        // from the fallback default, so the test would pass even if the
        // legacy file were never read.
        assert_eq!(s.max_seq_length, 96);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn legacy_sentence_distilbert_config_is_last_resort() {
        let root = tmpdir("legacy-root");
        let tdir = root.join("0_X");
        fs::create_dir_all(&tdir).unwrap();
        write(
            &tdir,
            "sentence_distilbert_config.json",
            r#"{"max_seq_length": 96, "do_lower_case": true}"#,
        );
        let s = resolve_tokenizer_settings(&tdir, &root);
        assert!(s.do_lower_case);
        assert_eq!(s.max_seq_length, 96);
        fs::remove_dir_all(&root).ok();
    }
}
