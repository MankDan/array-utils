#ifndef ARRAY_UTILS_HPP
#define ARRAY_UTILS_HPP

#include <cstddef>
#include <algorithm>
#include <cstddef>
#include <immintrin.h>
#include <intrin.h>

namespace array_utils {

//---------- CHECK AVX SUPPORT ----------// 

    inline bool is_avx_supported() {
        int cpuInfo[4] = { 0 };
        __cpuid(cpuInfo, 1);
        return (cpuInfo[2] & (1 << 28)) != 0;
    }
    inline bool is_avx512_supported() {
        int cpuInfo[4] = { 0 };
        __cpuid(cpuInfo, 7);
        return (cpuInfo[1] & (1 << 16)) != 0;
    }


//---------- AVX 128 OPTIMIZATION FOR PRIMITIVE TYPES ----------//

        // SIMD optimization for int (AVX 128) 
    inline size_t count_avx_128(const int* start, const int* end, const int& value) noexcept {
        size_t counter = 0;

        __m128i target = _mm_set1_epi32(value);
        const int* aligned_end = start + ((end - start) >> 2) * 4;

        for (; start < aligned_end; start += 4) {
            __m128i chunk = _mm_loadu_si128((const __m128i*) start);
            __mmask8 cmp = _mm_cmpeq_epi32_mask(chunk, target);
            counter += _mm_popcnt_u32(cmp);
        }
        while (start < end) {
            counter += *(start++) == value;
        }

        return counter;
    }

    // SIMD optimization for char (AVX 128) 
    inline size_t count_avx_128(const char* start, const char* end, const char& value) noexcept {
        size_t counter = 0;

        __m128i target = _mm_set1_epi8(value);
        const char* aligned_end = start + ((end - start) >> 4) * 16;

        for (; start < aligned_end; start += 16) {
            __m128i chunk = _mm_loadu_epi8((const __m128i*) start);
            __mmask16 cmp = _mm_cmpeq_epi8_mask(chunk, target);
            counter += _mm_popcnt_u32(cmp);
        }

        while (start < end) {
            counter += *(start++) == value;
        }

        return counter;
    }

    // SIMD optimization for float (AVX 128) 
    inline size_t count_avx_128(const float* start, const float* end, const float& value) noexcept {
        size_t counter = 0;

        __m128 target = _mm_set1_ps(value);
        const float* aligned_end = start + ((end - start) >> 2) * 4;

        for (; start < aligned_end; start += 4) {
            __m128 chunk = _mm_loadu_ps(start);
            __m128 cmp = _mm_cmpeq_ps(chunk, target);
            counter += _mm_popcnt_u32(_mm_movemask_ps(cmp));
        }

        while (start < end) {
            counter += *(start++) == value;
        }

        return counter;
    }

    // SIMD optimization for double (AVX 128)
    inline size_t count_avx_128(const double* start, const double* end, const double& value) noexcept {
        size_t counter = 0;

        __m128d target = _mm_set1_pd(value);
        const double* aligned_end = start + ((end - start) >> 1) * 2;

        for (; start < aligned_end; start += 2) {
            __m128d chunk = _mm_loadu_pd(start);
            __m128d cmp = _mm_cmpeq_pd(chunk, target);
            counter += _mm_popcnt_u32(_mm_movemask_pd(cmp));
        }

        while (start < end) {
            counter += *(start++) == value;
        }

        return counter;
    }

//---------- AVX 512 OPTIMIZATION FOR PRIMITIVE TYPES ----------//

        // SIMD optimization for int (AVX 512) 
    inline size_t count_avx_512(const int* start, const int* end, const int& value) noexcept {
        size_t counter = 0;

        __m512i target = _mm512_set1_epi32(value);
        const int* aligned_end = start + ((end - start) >> 4) * 16;

        for (; start < aligned_end; start += 16) {
            __m512i chunk = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(start));
            __mmask16 cmp = _mm512_cmpeq_epi32_mask(chunk, target);
            counter += _mm_popcnt_u32(cmp);
        }

        while (start < end) {
            counter += *(start++) == value;
        }

        return counter;
    }

    // SIMD optimization for char (AVX-512) 
    inline size_t count_avx_512(const char* start, const char* end, const char& value) noexcept {
        size_t counter = 0;

        __m512i target = _mm512_set1_epi8(value);
        const char* aligned_end = start + ((end - start) >> 6) * 64;

        for (; start < aligned_end; start += 64) {
            __m512i chunk = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(start));
            __mmask64 cmp = _mm512_cmpeq_epi8_mask(chunk, target);

            counter += _mm_popcnt_u64(cmp);
        }

        while (start < end) {
            counter += *(start++) == value;
        }

        return counter;
    }

    // SIMD optimization for float (AVX 512)
    inline size_t count_avx_512(const float* start, const float* end, const float& value) noexcept {
        size_t counter = 0;

        __m512 target = _mm512_set1_ps(value);

        const float* aligned_end = start + ((end - start) >> 4) * 16;

        for (; start < aligned_end; start += 16) {
            __m512 chunk = _mm512_loadu_ps(start);

            __mmask16 cmp = _mm512_cmpeq_epi32_mask(
                _mm512_castps_si512(chunk),
                _mm512_castps_si512(target)
            );

            counter += _mm_popcnt_u32(cmp);
        }

        while (start < end) {
            counter += (*(start++) == value);
        }

        return counter;
    }

    // SIMD optimization for double (AVX 512)
    inline size_t count_avx_512(const double* start, const double* end, const double& value) noexcept {
        size_t counter = 0;

        __m512d target = _mm512_set1_pd(value);

        const double* aligned_end = start + ((end - start) >> 3) * 8;

        for (; start < aligned_end; start += 8) {
            __m512d chunk = _mm512_loadu_pd(start);
            __mmask8 cmp = _mm512_cmpeq_epi64_mask(
                _mm512_castpd_si512(chunk),
                _mm512_castpd_si512(target)
            );
            counter += _mm_popcnt_u32(cmp);
        }

        while (start < end) {
            counter += (*(start++) == value);
        }

        return counter;
    }

    //---------- COUNT FUNCTIONS ----------//

    template <typename T>
    inline size_t count(const T* start, const T* end, const T& value) noexcept {
        return std::count(start, end, value);
    }

    inline size_t count(const int* start, const int* end, const int& value) {
        if (is_avx512_supported()) {
            return count_avx_512(start, end, value);
        }
        else if (is_avx_supported()) {
            return count_avx_128(start, end, value);
        }
        return std::count(start, end, value);
    }

    inline size_t count(const char* start, const char* end, const char& value) {
        if (is_avx512_supported()) {
            return count_avx_512(start, end, value);
        }
        else if (is_avx_supported()) {
            return count_avx_128(start, end, value);
        }
        return std::count(start, end, value);
    }

    inline size_t count(const float* start, const float* end, const float& value) {
        if (is_avx512_supported()) {
            return count_avx_512(start, end, value);
        }
        else if (is_avx_supported()) {
            return count_avx_128(start, end, value);
        }
        return std::count(start, end, value);
    }

    inline size_t count(const double* start, const double* end, const double& value) {
        if (is_avx512_supported()) {
            return count_avx_512(start, end, value);
        }
        else if (is_avx_supported()) {
            return count_avx_128(start, end, value);
        }
        return std::count(start, end, value);
    }





} // namespace array_utils

#endif // ARRAY_UTILS_HPP
