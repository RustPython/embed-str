//! Short string embedding for std `str`

/// Replacement of Box<[std::str::str]> for short string embedding
///
/// When string size is smaller than `std::mem::size_of::<usize>*2-1`,
/// embed the string content into itself rather than holding the pointer.
pub struct EmbeddingStr([usize; 2]);

const STR_INNER_SIZE: usize = std::mem::size_of::<EmbeddingStr>();

#[derive(Eq, PartialEq, Debug)]
pub enum EmbeddingStrMode {
    Boxed,
    Embedded,
}

impl EmbeddingStr {
    pub fn mode(&self) -> EmbeddingStrMode {
        // SAFETY: std::mem::align_of::<&str>() > 1
        if (self.0[0] & 1) == 0 {
            EmbeddingStrMode::Boxed
        } else {
            EmbeddingStrMode::Embedded
        }
    }

    pub fn as_str(&self) -> &str {
        match self.mode() {
            EmbeddingStrMode::Boxed => unsafe { std::mem::transmute(self.0) },
            EmbeddingStrMode::Embedded => {
                let embedded = unsafe { &*(self as *const Self as *const EmbeddedStr) };
                embedded.as_str()
            }
        }
    }
}

impl Drop for EmbeddingStr {
    fn drop(&mut self) {
        match self.mode() {
            EmbeddingStrMode::Boxed => {
                let _boxed: Box<str> = unsafe { std::mem::transmute(self.0) };
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
    #[inline]
    fn from(s: String) -> Self {
        const MAX_SIZE: usize = STR_INNER_SIZE - 1;
        // when size=0, it already has embedded form
        Self(if (1..=MAX_SIZE).contains(&s.len()) {
            let embedded = EmbeddedStr::from(s.as_str());
            unsafe { std::mem::transmute(embedded) }
        } else {
            let boxed = s.into_boxed_str();
            unsafe { std::mem::transmute(boxed) }
        })
    }
}

impl From<&'static str> for EmbeddingStr {
    #[inline]
    fn from(s: &str) -> Self {
        const MAX_SIZE: usize = STR_INNER_SIZE - 1;
        // when size=0, it already has embedded form
        Self(if (1..=MAX_SIZE).contains(&s.len()) {
            let embedded = EmbeddedStr::from(s);
            unsafe { std::mem::transmute(embedded) }
        } else {
            let boxed = s.to_owned().into_boxed_str();
            unsafe { std::mem::transmute(boxed) }
        })
    }
}

struct EmbeddedStr([u8; STR_INNER_SIZE]);

impl EmbeddedStr {
    fn as_str(&self) -> &str {
        let ptr;
        let encoded_len;
        #[cfg(target_endian = "little")]
        unsafe {
            ptr = self.0.as_ptr().offset(1);
            encoded_len = (*self.0.as_ptr()) as usize;
        }
        #[cfg(target_endian = "big")]
        unsafe {
            ptr = self.0.as_ptr();
            encoded_len = *self.0.as_ptr().offset(STR_INNER_SIZE - 1);
        }
        let pair = [ptr as usize, encoded_len as usize >> 1];
        unsafe { std::mem::transmute(pair) }
    }
}

impl From<&str> for EmbeddedStr {
    #[inline]
    fn from(s: &str) -> Self {
        debug_assert!(s.len() < STR_INNER_SIZE);
        let mut new = std::mem::MaybeUninit::<Self>::uninit();
        let mut_ptr = new.as_mut_ptr() as *mut u8;
        let encoded_len = (s.len() << 1) as u8 + 1;
        unsafe {
            #[cfg(target_endian = "little")]
            {
                std::ptr::copy_nonoverlapping(s.as_ptr(), mut_ptr.offset(1), s.len());
                mut_ptr.write(encoded_len);
            }
            #[cfg(target_endian = "big")]
            {
                std::ptr::copy_nonoverlapping(s.as_ptr(), mut_ptr, s.len());
                mut_ptr.offset(MAX_SIZE).write(encoded_len);
            }
            new.assume_init()
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
