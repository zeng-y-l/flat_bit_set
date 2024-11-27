pub mod biop;
mod iter;
#[cfg(test)]
mod proptest;

use crate::*;
use alloc::vec::Vec;
use biop::*;
use core::{cell::Cell, cmp::Ordering, convert::Infallible, fmt, ops};
pub use iter::GenericIter;

/// 键
pub trait Key: Copy + Default + Ord + 'static {
    /// 拆开键的高、低位
    ///
    /// 低位必须小于 `C::BITS`：
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let key = 114514;
    /// let (_, lo) = key.split::<u64>();
    /// assert!(lo < u64::BITS);
    /// ```
    ///
    /// 原先大小如何，拆开后依然不变：
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let k1 = 114514;
    /// let k2 = 1919810;
    /// assert_eq!(k1.cmp(&k2), k1.split::<u64>().cmp(&k2.split::<u64>()));
    /// ```
    fn split<C: Chunk>(self) -> (Self, u32);

    /// 合并键的高、低位
    ///
    /// 怎么拆的就怎么合并回去：
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let key = 114514;
    /// let (hi, lo) = key.split::<u64>();
    /// assert_eq!(key, hi.merge::<u64>(lo));
    /// ```
    #[must_use]
    fn merge<C: Chunk>(self, bit: u32) -> Self;
}

/// 块
///
/// 用于按位压缩存储一些布尔值。
pub trait Chunk:
    Copy
    + Eq
    + ops::Not<Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::BitOr<Output = Self>
    + ops::BitXor<Output = Self>
    + 'static
{
    /// 能存储的位数
    ///
    /// 必须是二的幂。
    const BITS: u32;

    /// 零
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// assert_eq!(u64::ZERO.count_ones(), 0);
    /// ```
    const ZERO: Self;

    /// 单个位
    ///
    /// 要求输入 bit 小于 [`Chunk::BITS`]；输出仅有一位且尾随 bit 个零。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let bit = 5;
    /// let mask = u64::mask(bit);
    /// assert_eq!(mask.count_ones(), 1);
    /// assert_eq!(mask.highest_bit(), bit);
    /// assert_eq!(mask.lowest_bit(), bit);
    /// ```
    fn mask(bit: u32) -> Self;

    /// 计算位的数量
    fn count_ones(self) -> u32;

    /// 最前一个 bit
    fn highest_bit(self) -> u32;

    /// 最后一个 bit
    fn lowest_bit(self) -> u32;

    /// 删除最后一位
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let c = 15532u64;
    /// let mut c2 = c;
    /// c2.remove_lowest_bit();
    /// assert_eq!(c2, c ^ u64::mask(c.trailing_zeros()));
    /// assert_eq!(c2.count_ones(), c.count_ones() - 1);
    /// ```
    fn remove_lowest_bit(&mut self);
}

/// 有序、稀疏、节省空间的 bitset，适用于小数据量
///
/// 实际上可以看作从 [`Key`] 到 [`Chunk`] 的 flat map。
/// 故而，其时间复杂度与 flat map 同：增删 O(n)、查询 O(log(n))、遍历 O(n)。
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct FlatBitSet<K = usize, C = u64>(Vec<(K, C)>);

/// 惰性集合
///
/// 可以进行惰性的二元逻辑运算，中间不计算、不分配内存，最后一次性迭代或收集。
///
/// 通过 [`FlatBitSet::thunk`] 创建。
///
/// ```
/// # use flat_bit_set::*;
/// let s1 = FlatBitSet::<usize>::from_iter([1, 2, 3]);
/// let s2 = FlatBitSet::<usize>::from_iter([2, 3, 4]);
/// let s3 = FlatBitSet::<usize>::from_iter([3, 4, 5]);
/// let thunk = (s1.thunk() ^ s2.thunk()) | (s1.thunk() & s3.thunk());
/// assert!(thunk.clone().into_iter().eq([1, 3, 4]));
/// assert_eq!(thunk.collect(), FlatBitSet::<usize>::from_iter([1, 3, 4]));
/// ```
#[must_use = "thunks are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct Thunk<I>(I);

/// 集合中元素的借用迭代器
///
/// 通过 [`FlatBitSet::iter`] 创建。有序。
pub type Iter<'a, K, C> = GenericIter<IterChunk<'a, K, C>, K, C>;
type IterChunk<'a, K, C> = core::iter::Copied<core::slice::Iter<'a, (K, C)>>;

/// 集合中元素的迭代器
///
/// 通过 [`FlatBitSet::into_iter`] 创建。有序。
pub type IntoIter<K, C> = GenericIter<IntoIterChunk<K, C>, K, C>;
type IntoIterChunk<K, C> = alloc::vec::IntoIter<(K, C)>;

fn split_key<K: Key, C: Chunk>(key: K) -> (K, C) {
    let (key, bit) = key.split::<C>();
    (key, C::mask(bit))
}

impl<K, C> FlatBitSet<K, C> {
    /// 空集
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let set = FlatBitSet::<usize>::new();
    /// assert!(set.is_empty());
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// 至少具有指定容量的块的空集
    ///
    /// 注意：所给的数字代表块数，而非元素数。块数可能远小于元素数。
    /// n 个元素最少需要 n / `C::BITS` 个块，最多需要 n 个。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let set = FlatBitSet::<usize>::with_capacity(10);
    /// assert!(set.is_empty());
    /// assert!(set.inner().capacity() >= 10);
    /// ```
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// 判断是否为空
    ///
    /// 相当于 `set.count() == 0`。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let mut set = FlatBitSet::<usize>::new();
    /// assert!(set.is_empty());
    /// set.insert(1);
    /// assert!(!set.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// 获取内部数组
    #[must_use]
    pub fn inner(&self) -> &Vec<(K, C)> {
        &self.0
    }

    /// 清空集合
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let mut set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// set.clear();
    /// assert!(set.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<K: Key, C: Chunk> FlatBitSet<K, C> {
    /// 元素个数
    ///
    /// 注意：时间复杂度是 O(n)！
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// assert_eq!(set.count(), 3);
    /// ```
    #[must_use]
    pub fn count(&self) -> usize {
        self.iter().count()
    }

    /// 判断元素是否属于集合
    ///
    /// 可以使用 `set[key]`，作用相同，更简洁。
    ///
    /// 时间复杂度：O(log(n))。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// assert_eq!(set.contains(1), true);
    /// assert_eq!(set[114514], false);
    /// ```
    #[must_use]
    #[inline]
    pub fn contains(&self, key: K) -> bool {
        let (key, mask) = split_key(key);
        let idx = self.0.binary_search_by_key(&key, |c| c.0);
        idx.is_ok_and(|idx| self.0[idx].1 & mask != C::ZERO)
    }

    fn search_mut(&mut self, key: K) -> (K, C, Result<(usize, &mut C), usize>, bool) {
        let (key, mask) = split_key(key);
        let (res, old) = match self.0.binary_search_by_key(&key, |c| c.0) {
            Ok(idx) => {
                let old = self.0[idx].1 & mask != C::ZERO;
                (Ok((idx, &mut self.0[idx].1)), old)
            }
            Err(idx) => (Err(idx), false),
        };
        (key, mask, res, old)
    }

    /// 插入元素
    ///
    /// 若元素是新的，返回 `true`（与标准库相同）。若已有元素，集合不变。
    ///
    /// 时间复杂度：O(n)。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let mut set = FlatBitSet::<usize>::new();
    /// assert_eq!(set.insert(1), true);
    /// assert_eq!(set.insert(1), false);
    /// ```
    pub fn insert(&mut self, key: K) -> bool {
        let (key, mask, res, old) = self.search_mut(key);
        match res {
            Ok((_, chunk)) => *chunk = *chunk | mask,
            Err(idx) => self.0.insert(idx, (key, mask)),
        }
        !old
    }

    /// 删除元素
    ///
    /// 若已有元素，返回 `true`（与标准库相同）。若没有元素，集合不变。
    ///
    /// 时间复杂度：O(n)。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let mut set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// assert_eq!(set.remove(1), true);
    /// assert_eq!(set.remove(1), false);
    /// ```
    pub fn remove(&mut self, key: K) -> bool {
        let (_, mask, res, old) = self.search_mut(key);
        if let Ok((idx, chunk)) = res {
            *chunk = *chunk & !mask;
            if *chunk == C::ZERO {
                self.0.remove(idx);
            }
        }
        old
    }

    /// 翻转某一位
    ///
    /// 若已有元素，删除之；否则插入之。
    ///
    /// 若已有元素，则返回 `true`。
    ///
    /// 时间复杂度：O(n)。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let mut set = FlatBitSet::<usize>::new();
    /// assert_eq!(set.toggle(1), false);
    /// assert_eq!(set[1], true);
    /// assert_eq!(set.toggle(1), true);
    /// assert_eq!(set[1], false);
    /// ```
    pub fn toggle(&mut self, key: K) -> bool {
        let (key, mask, res, old) = self.search_mut(key);
        match res {
            Ok((idx, chunk)) => {
                *chunk = *chunk ^ mask;
                if *chunk == C::ZERO {
                    self.0.remove(idx);
                }
            }
            Err(idx) => self.0.insert(idx, (key, mask)),
        }
        old
    }

    /// 设置某一位
    ///
    /// 设置某元素的存在性。若 `presence` 为 `true` 则插入之；否则删除之。
    /// 若已有元素而插入，或没有元素而删除，集合均不变。
    ///
    /// 若已有元素，返回 `true`。
    ///
    /// 时间复杂度：O(n)。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let mut set = FlatBitSet::<usize>::new();
    /// assert_eq!(set.set(1, true), false);
    /// assert_eq!(set[1], true);
    /// assert_eq!(set.set(1, false), true);
    /// assert_eq!(set[1], false);
    /// ```
    pub fn set(&mut self, key: K, presence: bool) -> bool {
        let (key, mask, res, old) = self.search_mut(key);
        match res {
            Ok((idx, chunk)) => {
                if presence != old {
                    *chunk = *chunk ^ mask;
                }
                if *chunk == C::ZERO {
                    self.0.remove(idx);
                }
            }
            Err(idx) => {
                if presence {
                    self.0.insert(idx, (key, mask));
                }
            }
        }
        old
    }

    /// 尽可能缩小容量
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let mut set = FlatBitSet::<usize>::with_capacity(10);
    /// set.extend([1, 2, 3]);
    /// assert!(set.inner().capacity() >= 10);
    /// set.shrink_to_fit();
    /// assert!(set.inner().capacity() >= 1);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }

    /// 保留至少指定容量的块
    ///
    /// 注意：所给的数字代表块数，而非元素数。块数可能远小于元素数。
    /// n 个元素最少需要 n / `C::BITS` 个块，最多需要 n 个。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let mut set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// set.reserve(10);
    /// assert!(set.inner().capacity() >= 11);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    /// 判断子集
    ///
    /// 若该集合是另一个集合的子集（可以相等），返回 `true`。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// assert!(set.is_subset(&set));
    /// assert!(!set.is_subset(&FlatBitSet::from_iter([2, 3, 4])));
    /// assert!(set.is_subset(&FlatBitSet::from_iter([1, 2, 3, 4])));
    /// ```
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.try_zip_fold_chunks(
            other,
            (),
            |(), _, _| Err(()),
            |(), _, _| Ok(()),
            |(), _, a, b| if a & b == a { Ok(()) } else { Err(()) },
        )
        .is_ok()
    }

    /// 判断超集
    ///
    /// 若该集合是另一个集合的超集（可以相等），返回 `true`。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// assert!(set.is_superset(&set));
    /// assert!(!set.is_superset(&FlatBitSet::from_iter([2, 3, 4])));
    /// assert!(set.is_superset(&FlatBitSet::from_iter([2, 3])));
    /// ```
    #[must_use]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// 比较子集关系
    ///
    /// 若该集合是另一个集合的真子集，返回 `Some(Less)`；
    /// 若是真超集，返回 `Some(Greater)`；
    /// 若相等，返回 `Some(Equal)`；否则返回 `None`。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// # use std::cmp::Ordering::*;
    /// let set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// assert_eq!(set.subset_cmp(&set), Some(Equal));
    /// assert_eq!(set.subset_cmp(&FlatBitSet::from_iter([1, 2, 3, 4])), Some(Less));
    /// assert_eq!(set.subset_cmp(&FlatBitSet::from_iter([1, 2])), Some(Greater));
    /// assert_eq!(set.subset_cmp(&FlatBitSet::from_iter([2, 3, 4])), None);
    /// ```
    #[must_use]
    pub fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        let combine = |o1, o2| match (o1, o2) {
            (o, Ordering::Equal) | (Ordering::Equal, o) => Ok(o),
            _ if o1 == o2 => Ok(o2),
            _ => Err(()),
        };
        let cmp = |a, b| match () {
            () if a == b => Ok(Ordering::Equal),
            () if a & b == a => Ok(Ordering::Less),
            () if a & b == b => Ok(Ordering::Greater),
            () => Err(()),
        };
        self.try_zip_fold_chunks(
            other,
            Ordering::Equal,
            |ord, _, _| combine(ord, Ordering::Greater),
            |ord, _, _| combine(ord, Ordering::Less),
            |ord, _, a, b| combine(ord, cmp(a, b)?),
        )
        .ok()
    }

    /// 判断交集为空
    ///
    /// 若两集合没有共同元素，返回 `true`。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let set = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// assert!(set.is_disjoint(&FlatBitSet::from_iter([4, 5, 6])));
    /// assert!(!set.is_disjoint(&FlatBitSet::from_iter([2, 3, 4])));
    /// ```
    #[must_use]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.try_zip_fold_chunks(
            other,
            (),
            |(), _, _| Ok(()),
            |(), _, _| Ok(()),
            |(), _, a, b| if a & b == C::ZERO { Ok(()) } else { Err(()) },
        )
        .is_ok()
    }

    fn try_zip_fold_chunks<T, E>(
        &self,
        other: &Self,
        init: T,
        mut first: impl FnMut(T, K, C) -> Result<T, E>,
        mut second: impl FnMut(T, K, C) -> Result<T, E>,
        mut both: impl FnMut(T, K, C, C) -> Result<T, E>,
    ) -> Result<T, E> {
        let mut it2 = other.chunks().peekable();
        let acc = self.chunks().try_fold(init, |acc, (k1, c1)| {
            let acc = core::iter::from_fn(|| it2.next_if(|&(k2, _)| k2 < k1))
                .try_fold(acc, |acc, (k2, c2)| second(acc, k2, c2))?;
            match it2.next_if(|&(k2, _)| k1 == k2) {
                Some((_, c2)) => both(acc, k1, c1, c2),
                None => first(acc, k1, c1),
            }
        })?;
        it2.try_fold(acc, |acc, (k2, c2)| second(acc, k2, c2))
    }

    /// 原地进行二元逻辑运算
    ///
    /// 请使用 [`FlatBitSet::intersection_with`] 之类的方法。
    /// 也可以使用 `s1 & &s2` 之类的运算符，作用相同，更为简洁。
    /// 运算符会获取左侧集合的所有权，改写后返回其自身。
    ///
    /// 会预先计算大小，最多只会分配一次。不会减少容量。
    ///
    /// 也可以看看创造新集合的 [`FlatBitSet::intersection`]、
    /// 返回迭代器的  [`FlatBitSet::intersection_iter`]
    /// 以及惰性完成的 [`Thunk::intersection`]
    #[allow(clippy::missing_panics_doc)]
    pub fn perform_biop_with<O: BiOp<C>>(&mut self, other: &Self) {
        let mid = self.0.len();
        let cap = match self.try_zip_fold_chunks::<_, Infallible>(
            other,
            mid,
            |n, _, _| Ok(n),
            |n, _, c2| Ok(n + usize::from(O::second(c2) != C::ZERO)),
            |n, _, _, _| Ok(n),
        ) {
            Ok(cap) => cap,
            Err(x) => match x {},
        };
        self.0.resize(cap, (K::default(), C::ZERO));
        let data = Cell::from_mut(&mut self.0[..]).as_slice_of_cells();
        let mut writer = data.iter().rev();
        let mut write = |k, c| {
            if c != C::ZERO {
                writer.next().unwrap().set((k, c));
            }
        };
        let mut it2 = other.chunks().rev().peekable();
        for kv in data[0..mid].iter().rev() {
            let (k1, c1) = kv.get();
            while let Some((k2, c2)) = it2.next_if(|&(k2, _)| k2 > k1) {
                write(k2, O::second(c2));
            }
            let cur = it2
                .next_if(|&(k2, _)| k1 == k2)
                .map_or_else(|| O::first(c1), |(_, c2)| O::both(c1, c2));
            write(k1, cur);
        }
        for (k2, c2) in it2 {
            write(k2, O::second(c2));
        }
        let first = writer.len();
        self.0.drain(..first);
    }

    fn chunks(&self) -> IterChunk<K, C> {
        self.0.iter().copied()
    }

    /// 集合中元素的借用迭代器
    ///
    /// 有序。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let set = FlatBitSet::<usize>::from_iter([3, 2, 1]);
    /// assert!(set.iter().eq([1, 2, 3]));
    /// ```
    pub fn iter(&self) -> Iter<K, C> {
        Iter::new(self.chunks())
    }

    /// 获取惰性集合
    ///
    /// 请参阅 [`Thunk`]。
    pub fn thunk(&self) -> Thunk<IterChunk<K, C>> {
        Thunk(self.chunks())
    }
}

impl<K, C> Default for FlatBitSet<K, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key + fmt::Debug, C: Chunk> fmt::Debug for FlatBitSet<K, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self).finish()
    }
}

impl<K: Key, C: Chunk> FromIterator<K> for FlatBitSet<K, C> {
    fn from_iter<T: IntoIterator<Item = K>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut this = Self::with_capacity(iter.size_hint().0 / C::BITS as usize);
        this.extend(iter);
        this
    }
}

impl<K: Key, C: Chunk> Extend<K> for FlatBitSet<K, C> {
    fn extend<T: IntoIterator<Item = K>>(&mut self, iter: T) {
        for key in iter {
            self.insert(key);
        }
    }
}

impl<'a, K: Key, C: Chunk> IntoIterator for &'a FlatBitSet<K, C> {
    type Item = K;
    type IntoIter = Iter<'a, K, C>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Key, C: Chunk> IntoIterator for FlatBitSet<K, C> {
    type Item = K;
    type IntoIter = IntoIter<K, C>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.0.into_iter())
    }
}

impl<K: Key, C: Chunk> ops::Index<K> for FlatBitSet<K, C> {
    type Output = bool;

    fn index(&self, index: K) -> &Self::Output {
        if self.contains(index) {
            &true
        } else {
            &false
        }
    }
}

impl<I, K, C> Thunk<I>
where
    I: Iterator<Item = (K, C)>,
    K: Key,
    C: Chunk,
{
    /// 进行惰性二元运算
    ///
    /// 请使用 [`Thunk::intersection`] 之类的方法。
    /// 也可以使用 `t1 & t2` 之类的运算符，作用相同，更为简洁。
    pub fn perform_biop<O, I2>(self, other: Thunk<I2>) -> Thunk<BiOpIter<O, I, I2>>
    where
        O: BiOp<C>,
        I2: Iterator<Item = (K, C)>,
    {
        Thunk(BiOpIter::new(self.0, other.0))
    }

    /// 转换为 [`FlatBitSet`]
    ///
    /// 会预先计算大小，最多只会分配一次。
    ///
    /// ```
    /// # use flat_bit_set::*;
    /// let s1 = FlatBitSet::<usize>::from_iter([1, 2, 3]);
    /// let s2 = FlatBitSet::<usize>::from_iter([2, 3, 4]);
    /// let thunk = s1.thunk() ^ s2.thunk();
    /// assert_eq!(thunk.collect(), FlatBitSet::<usize>::from_iter([1, 4]));
    /// ```
    pub fn collect(self) -> FlatBitSet<K, C>
    where
        I: Clone,
    {
        let len = self.0.clone().count();
        let mut out = FlatBitSet::with_capacity(len);
        for (k, c) in self.0 {
            out.0.push((k, c));
        }
        debug_assert_eq!(out.0.len(), len);
        out
    }
}

impl<I, K, C> IntoIterator for Thunk<I>
where
    I: Iterator<Item = (K, C)>,
    K: Key,
    C: Chunk,
{
    type Item = K;

    type IntoIter = GenericIter<I, K, C>;

    fn into_iter(self) -> Self::IntoIter {
        GenericIter::new(self.0)
    }
}

macro_rules! impl_biop {
    (
        $trait:ident { $fn:ident },
        $traitasgn:ident { $fnasgn:ident },
        $opc:literal,
        $ops:literal,
        $res:literal,
        $mthd:ident,
        $mthditer:ident,
        $mthdwith:ident,
        $biop:ty
    ) => {
        impl<I, K, C> Thunk<I>
        where
            I: Iterator<Item = (K, C)>,
            K: Key,
            C: Chunk,
        {
            #[doc = concat!("惰性计算", $opc)]
            ///
            #[doc = concat!("可以使用运算符 `t1 ", $ops, " t2`，作用相同，更为简洁。")]
            ///
            /// ```
            /// # use flat_bit_set::*;
            /// let s1 = FlatBitSet::<usize>::from_iter([1, 2, 3]);
            /// let s2 = FlatBitSet::<usize>::from_iter([2, 3, 4]);
            #[doc = concat!("let thunk = s1.thunk().", stringify!($mthd), "(s2.thunk());")]
            #[doc = concat!("assert!(thunk.into_iter().eq(", $res, "));")]
            /// ```
            ///
            #[doc = concat!("可以看看返回新集合的 [`FlatBitSet::", stringify!($mthd), "`]、就地计算的 [`FlatBitSet::", stringify!($mthdwith), "`] 以及返回迭代器的 [`FlatBitSet::", stringify!($mthditer), "`]。")]
            pub fn $mthd<I2: Iterator<Item = (K, C)>>(
                self,
                other: Thunk<I2>,
            ) -> Thunk<BiOpIter<$biop, I, I2>> {
                self.perform_biop(other)
            }
        }

        impl<K: Key, C: Chunk> FlatBitSet<K, C> {
            #[doc = concat!("计算", $opc)]
            ///
            #[doc = concat!("可以使用运算符 `&s1 ", $ops, " &s2`，作用相同，更为简洁。")]
            ///
            /// ```
            /// # use flat_bit_set::*;
            /// let s1 = FlatBitSet::<usize>::from_iter([1, 2, 3]);
            /// let s2 = FlatBitSet::<usize>::from_iter([2, 3, 4]);
            #[doc = concat!("let set = s1.", stringify!($mthd), "(&s2);")]
            #[doc = concat!("assert!(set.iter().eq(", $res, "));")]
            /// ```
            ///
            #[doc = concat!("可以看看就地计算的 [`FlatBitSet::", stringify!($mthdwith), "`]、返回迭代器的 [`FlatBitSet::", stringify!($mthditer), "`] 以及惰性计算的 [`Thunk::", stringify!($mthd), "`]（该方法在内部就是用了它）。")]
            #[must_use]
            pub fn $mthd(&self, other: &Self) -> Self {
                self.thunk().$mthd(other.thunk()).collect()
            }

            #[doc = concat!("就地计算", $opc)]
            ///
            #[doc = concat!("可以使用运算符 `s1 ", $ops, " &s2`，作用相同，更为简洁。")]
            /// 运算符会获取左侧集合的所有权，改写后返回其自身。
            ///
            /// ```
            /// # use flat_bit_set::*;
            /// let mut s1 = FlatBitSet::<usize>::from_iter([1, 2, 3]);
            /// let s2 = FlatBitSet::<usize>::from_iter([2, 3, 4]);
            #[doc = concat!("s1.", stringify!($mthdwith), "(&s2);")]
            #[doc = concat!("assert!(s1.iter().eq(", $res, "));")]
            /// ```
            ///
            #[doc = concat!("可以看看返回新集合的 [`FlatBitSet::", stringify!($mthd), "`]、返回迭代器的 [`FlatBitSet::", stringify!($mthditer), "`] 以及惰性计算的 [`Thunk::", stringify!($mthd), "`]。")]
            pub fn $mthdwith(&mut self, other: &Self) {
                self.perform_biop_with::<$biop>(other);
            }


            #[doc = concat!($opc, "中元素的迭代器")]
            ///
            /// ```
            /// # use flat_bit_set::*;
            /// let mut s1 = FlatBitSet::<usize>::from_iter([1, 2, 3]);
            /// let s2 = FlatBitSet::<usize>::from_iter([2, 3, 4]);
            #[doc = concat!("let iter = s1.", stringify!($mthditer), "(&s2);")]
            #[doc = concat!("assert!(iter.eq(", $res, "));")]
            /// ```
            ///
            #[doc = concat!("可以看看返回新集合的 [`FlatBitSet::", stringify!($mthd), "`]、就地计算的 [`FlatBitSet::", stringify!($mthdwith), "`] 以及惰性计算的 [`Thunk::", stringify!($mthd), "`]（该方法在内部就是用了它）。")]
            pub fn $mthditer<'a>(
                &'a self,
                other: &'a Self,
            ) -> GenericIter<BiOpIter<$biop, IterChunk<'a, K, C>, IterChunk<'a, K, C>>, K, C> {
                self.thunk().$mthd(other.thunk()).into_iter()
            }
        }

        impl<K: Key, C: Chunk> ops::$traitasgn<&Self> for FlatBitSet<K, C> {
            fn $fnasgn(&mut self, rhs: &Self) {
                self.$mthdwith(rhs);
            }
        }

        impl<K: Key, C: Chunk> ops::$trait<&Self> for FlatBitSet<K, C> {
            type Output = Self;

            fn $fn(mut self, rhs: &Self) -> Self::Output {
                self.$mthdwith(rhs);
                self
            }
        }

        impl<K: Key, C: Chunk> ops::$trait for &FlatBitSet<K, C> {
            type Output = FlatBitSet<K, C>;

            fn $fn(self, rhs: Self) -> Self::Output {
                self.$mthd(rhs)
            }
        }

        impl<I, I2, K, C> ops::$trait<Thunk<I2>> for Thunk<I>
        where
            I: Iterator<Item = (K, C)>,
            I2: Iterator<Item = (K, C)>,
            K: Key,
            C: Chunk,
        {
            type Output = Thunk<BiOpIter<$biop, I, I2>>;

            fn $fn(self, rhs: Thunk<I2>) -> Self::Output {
                self.$mthd(rhs)
            }
        }
    };
}

impl_biop!(
    BitAnd { bitand },
    BitAndAssign { bitand_assign },
    "交集",
    "&",
    "[2, 3]",
    intersection,
    intersection_iter,
    intersection_with,
    AndOp
);
impl_biop!(
    BitOr { bitor },
    BitOrAssign { bitor_assign },
    "并集",
    "|",
    "[1, 2, 3, 4]",
    union,
    union_iter,
    union_with,
    OrOp
);
impl_biop!(
    BitXor { bitxor },
    BitXorAssign { bitxor_assign },
    "对称差",
    "^",
    "[1, 4]",
    symmetric_difference,
    symmetric_difference_iter,
    symmetric_difference_with,
    XorOp
);
impl_biop!(
    Sub { sub },
    SubAssign { sub_assign },
    "差集",
    "-",
    "[1]",
    difference,
    difference_iter,
    difference_with,
    DiffOp
);

macro_rules! impl_key {
    ($ty:ty) => {
        #[allow(
            clippy::cast_lossless,
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss
        )]
        impl Key for $ty {
            #[inline]
            fn merge<C: Chunk>(self, bit: u32) -> Self {
                self | bit as Self
            }

            #[inline]
            fn split<C: Chunk>(self) -> (Self, u32) {
                let mask = (C::BITS - 1) as Self;
                (self & !mask, (self & mask) as u32)
            }
        }
    };
}

macro_rules! impl_chunk {
    ($ty:ty) => {
        impl Chunk for $ty {
            const BITS: u32 = Self::BITS;

            const ZERO: Self = 0;

            #[inline]
            fn mask(bit: u32) -> Self {
                1 << bit
            }

            #[inline]
            fn count_ones(self) -> u32 {
                self.count_ones()
            }

            #[inline]
            fn highest_bit(self) -> u32 {
                self.ilog2()
            }

            #[inline]
            fn lowest_bit(self) -> u32 {
                self.trailing_zeros()
            }

            #[inline]
            fn remove_lowest_bit(&mut self) {
                *self &= *self - 1;
            }
        }
    };
}

impl_key!(u8);
impl_key!(u16);
impl_key!(u32);
impl_key!(u64);
impl_key!(u128);
impl_key!(usize);
impl_key!(i8);
impl_key!(i16);
impl_key!(i32);
impl_key!(i64);
impl_key!(i128);
impl_key!(isize);

impl_chunk!(u32);
impl_chunk!(u64);
impl_chunk!(u128);
impl_chunk!(usize);
