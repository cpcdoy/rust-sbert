pub mod distilroberta;
pub mod sbert;
pub mod character_bert;

pub use self::distilroberta::DistilRobertaForSequenceClassification;
pub use self::sbert::SBert;
pub use self::character_bert::{CharacterBertModel, CharacterBertForSequenceClassification, CharacterBertConfig};

// Utils
pub fn pad_sort<O: Ord>(arr: &[O]) -> Vec<usize> {
    let mut idx = (0..arr.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| arr[i].cmp(&arr[j]));
    idx
}
