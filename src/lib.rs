//! Short string embedding for std `str`

use std::borrow::Cow;
use std::mem::{self, MaybeUninit};
use std::ptr;

/// Replacement of Box<[std::str::str]> for short string embedding
///
/// When string size is smaller than `std::mem::size_of::<usize>*2-1`,
/// embed the string content into itself rather than holding the pointer.
#[cfg_attr(target_pointer_width = "64", repr(align(8)))]
#[cfg_attr(target_pointer_width = "32", repr(align(4)))]
pub struct EmbeddingStr(MaybeUninit<[u8; STR_INNER_SIZE]>);

// Little Endian 32 bit:
// heap : |x|l|l|l|p|p|p|p|
// embed: |x|s|s|s|s|s|s|s|
// Big Endian 32 bit:
// heap : |p|p|p|p|l|l|l|x|
// embed: |s|s|s|s|s|s|s|x|
//
// x: discriminant byte; the first bit is 1 if embedded and 0 if heap, the rest of the
// byte is the len<<1
// l: the rest of the len<<1 if heap
// p: ptr to data if heap
// s: str data (1 utf8 byte) if embedded
//
// we can shift the len<<1 because the max slice len is actually isize::MAX as usize:
// https://stackoverflow.com/questions/32324794/maximum-size-of-an-array-in-32-bits

const STR_INNER_SIZE: usize = std::mem::size_of::<usize>() * 2;
const MAX_EMBEDDED_LEN: usize = STR_INNER_SIZE - 1;

#[derive(Eq, PartialEq, Debug)]
pub enum EmbeddingStrMode {
    Boxed,
    Embedded,
}

#[inline]
fn len_least_significant([len, ptr]: [usize; 2]) -> [usize; 2] {
    if cfg!(target_endian = "little") {
        [len, ptr]
    } else {
        [ptr, len]
    }
}

impl EmbeddingStr {
    fn new_embedded(s: &str) -> Self {
        debug_assert!(s.len() <= MAX_EMBEDDED_LEN);
        let mut new = std::mem::MaybeUninit::uninit();
        let mut_ptr = new.as_mut_ptr() as *mut u8;
        let encoded_len = ((s.len() as u8) << 1) | 1;
        unsafe {
            if cfg!(target_endian = "little") {
                std::ptr::copy_nonoverlapping(s.as_ptr(), mut_ptr.add(1), s.len());
                mut_ptr.write(encoded_len);
            } else {
                std::ptr::copy_nonoverlapping(s.as_ptr(), mut_ptr, s.len());
                mut_ptr.add(STR_INNER_SIZE - 1).write(encoded_len);
            }
        }
        Self(new)
    }

    fn new_heap(s: Box<str>) -> Self {
        let len = s.len();
        let ptr = Box::into_raw(s) as *mut u8 as usize;
        let inner = len_least_significant([len << 1, ptr]);
        Self(unsafe { mem::transmute(inner) })
    }

    // SAFETY: must be in fully initialized heap mode to call
    unsafe fn heap_ptr(&self) -> *const str {
        let inner = mem::transmute_copy(&self.0);
        let [len, ptr] = len_least_significant(inner);
        ptr::slice_from_raw_parts(ptr as *const u8, len >> 1) as *const str
    }

    fn embedded_len(&self) -> Option<usize> {
        // SAFETY: the least significant byte of the structure is always initialized
        let discriminant_byte = unsafe {
            if cfg!(target_endian = "little") {
                std::mem::transmute_copy::<_, u8>(&self.0)
            } else {
                self.0.as_ptr().cast::<u8>().add(STR_INNER_SIZE - 1).read()
            }
        };
        if discriminant_byte & 1 == 0 {
            None
        } else {
            Some(usize::from(discriminant_byte >> 1))
        }
    }

    pub fn mode(&self) -> EmbeddingStrMode {
        if self.embedded_len().is_some() {
            EmbeddingStrMode::Embedded
        } else {
            EmbeddingStrMode::Boxed
        }
    }

    pub fn as_str(&self) -> &str {
        match self.embedded_len() {
            None => unsafe { &*self.heap_ptr() },
            Some(len) => {
                let ptr = self.0.as_ptr().cast::<u8>();
                let start = if cfg!(target_endian = "little") {
                    unsafe { ptr.add(1) }
                } else {
                    ptr
                };
                let sptr = ptr::slice_from_raw_parts(start, len) as *const str;
                unsafe { &*sptr }
            }
        }
    }
}

impl Drop for EmbeddingStr {
    fn drop(&mut self) {
        match self.mode() {
            EmbeddingStrMode::Boxed => {
                let _boxed = unsafe { Box::from_raw(self.heap_ptr() as *mut str) };
            }
            EmbeddingStrMode::Embedded => {
                // nothing to do
            }
        }
    }
}

impl std::fmt::Display for EmbeddingStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.as_str(), f)
    }
}

impl std::fmt::Debug for EmbeddingStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}({:?})", self.mode(), self.as_str())
    }
}

impl From<String> for EmbeddingStr {
    #[inline(always)]
    fn from(s: String) -> Self {
        Self::from(Cow::Owned(s))
    }
}

impl From<&'_ str> for EmbeddingStr {
    #[inline(always)]
    fn from(s: &str) -> Self {
        EmbeddingStr::from(Cow::Borrowed(s))
    }
}

impl From<Cow<'_, str>> for EmbeddingStr {
    #[inline]
    fn from(s: Cow<'_, str>) -> Self {
        if s.len() <= MAX_EMBEDDED_LEN {
            Self::new_embedded(&s)
        } else {
            Self::new_heap(s.into_owned().into_boxed_str())
        }
    }
}

impl From<Box<str>> for EmbeddingStr {
    #[inline]
    fn from(s: Box<str>) -> Self {
        if s.len() <= MAX_EMBEDDED_LEN {
            Self::new_embedded(&s)
        } else {
            Self::new_heap(s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle() {
        assert_eq!(STR_INNER_SIZE, 16);

        let s = EmbeddingStr::from("");
        assert_eq!(s.mode(), EmbeddingStrMode::Embedded);
        assert_eq!(s.as_str(), "");
        let s = EmbeddingStr::from("a");
        assert_eq!(s.mode(), EmbeddingStrMode::Embedded);
        assert_eq!(s.as_str(), "a");
        let s = EmbeddingStr::from("abcdxyz01");
        assert_eq!(s.mode(), EmbeddingStrMode::Embedded);
        assert_eq!(s.as_str(), "abcdxyz01");
        let s = EmbeddingStr::from("123456789012345");
        assert_eq!(s.mode(), EmbeddingStrMode::Embedded);
        assert_eq!(s.as_str(), "123456789012345");
        let s = EmbeddingStr::from("1234567890123456");
        assert_eq!(s.mode(), EmbeddingStrMode::Boxed);
        assert_eq!(s.as_str(), "1234567890123456");
        let slong = EmbeddingStr::from("something longer than 15 byets");
        assert_eq!(slong.mode(), EmbeddingStrMode::Boxed);
        assert_eq!(slong.as_str(), "something longer than 15 byets");
    }

    #[test]
    fn test_format() {
        let s = EmbeddingStr::from("a");
        assert_eq!(format!("{}", s), "a".to_owned());
        assert_eq!(format!("{:?}", s), "Embedded(\"a\")".to_owned());
        let s = EmbeddingStr::from("1234567890123456");
        assert_eq!(format!("{}", s), "1234567890123456".to_owned());
        assert_eq!(format!("{:?}", s), "Boxed(\"1234567890123456\")".to_owned());
    }
}
