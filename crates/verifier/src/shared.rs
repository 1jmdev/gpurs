use std::{
    cell::Cell,
    collections::BTreeSet,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    rc::Rc,
};

use rustc_hash::FxHasher;

#[derive(Clone, Debug)]
pub struct Shared<T>(Rc<Data<T>>);

#[derive(Clone, Debug)]
struct Data<T> {
    value: T,
    hash: Cell<Option<u64>>,
}

impl<T> From<T> for Shared<T> {
    fn from(value: T) -> Self {
        Self(Rc::new(Data {
            value,
            hash: Cell::new(None),
        }))
    }
}

impl<T: Default> Default for Shared<T> {
    fn default() -> Self {
        T::default().into()
    }
}

impl<T> Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0.value
    }
}

impl<T: Clone> DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut T {
        let data = Rc::make_mut(&mut self.0);
        data.hash.set(None);
        &mut data.value
    }
}

impl<T: PartialEq> PartialEq for Shared<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
            || (self
                .0
                .hash
                .get()
                .zip(other.0.hash.get())
                .is_none_or(|(left, right)| left == right)
                && self.0.value == other.0.value)
    }
}

impl<T: Eq> Eq for Shared<T> {}

impl<T: Ord> Ord for Shared<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if Rc::ptr_eq(&self.0, &other.0) {
            std::cmp::Ordering::Equal
        } else {
            self.0.value.cmp(&other.0.value)
        }
    }
}

impl<T: Ord> PartialOrd for Shared<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Hash> Hash for Shared<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let hash = self.0.hash.get().unwrap_or_else(|| {
            let mut hasher = FxHasher::default();
            self.0.value.hash(&mut hasher);
            let hash = hasher.finish();
            self.0.hash.set(Some(hash));
            hash
        });
        state.write_u64(hash);
    }
}

impl<T: Clone> Shared<T> {
    pub fn make_mut(&mut self) -> &mut T {
        self
    }

    pub fn into_inner(self) -> T {
        Rc::unwrap_or_clone(self.0).value
    }
}

impl<T: Clone + IntoIterator> IntoIterator for Shared<T> {
    type Item = T::Item;
    type IntoIter = T::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_inner().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Shared<T>
where
    &'a T: IntoIterator,
{
    type Item = <&'a T as IntoIterator>::Item;
    type IntoIter = <&'a T as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0.value).into_iter()
    }
}

impl<A, T: FromIterator<A>> FromIterator<A> for Shared<T> {
    fn from_iter<I: IntoIterator<Item = A>>(iter: I) -> Self {
        T::from_iter(iter).into()
    }
}

impl<T: Ord, const N: usize> From<[T; N]> for Shared<BTreeSet<T>> {
    fn from(value: [T; N]) -> Self {
        BTreeSet::from(value).into()
    }
}
