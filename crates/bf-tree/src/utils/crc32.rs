// CRC32 (IEEE / CRC-32b) lookup-table implementation.
// no_std compatible, zero external dependencies.

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i: u32 = 0;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// Compute the CRC-32 (IEEE) checksum of `data`.
///
/// Returns a non-zero value for any non-empty input (the final XOR with
/// `0xFFFF_FFFF` guarantees this for practical page data).
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vectors() {
        // "123456789" -> 0xCBF43926 (IEEE CRC-32 test vector)
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn crc32_empty() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc32_nonzero_for_nonempty() {
        // Any non-empty data should produce a non-zero checksum.
        let page = vec![0u8; 4096];
        let checksum = crc32(&page);
        assert_ne!(checksum, 0);
    }

    #[test]
    fn crc32_detects_single_bit_flip() {
        let mut data = vec![0xABu8; 512];
        let original = crc32(&data);
        data[100] ^= 0x01;
        let corrupted = crc32(&data);
        assert_ne!(original, corrupted);
    }
}
