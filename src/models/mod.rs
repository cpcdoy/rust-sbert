pub mod character_bert;
pub mod distilroberta;
pub mod sbert;

pub use self::character_bert::{
    CharacterBertConfig, CharacterBertForSequenceClassification, CharacterBertModel,
};
pub use self::distilroberta::DistilRobertaForSequenceClassification;
pub use self::sbert::SBert;

// Utils
pub fn pad_sort<O: Ord>(arr: &[O]) -> Vec<usize> {
    let mut idx = (0..arr.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| arr[i].cmp(&arr[j]));
    idx
}
