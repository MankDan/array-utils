#ifndef ARRAY_UTILS_HPP
#define ARRAY_UTILS_HPP

#include <cstddef>

namespace array_utils {

	bool is_avx_supported();
	bool is_avx512_supported();

	template <typename T>
	size_t count(const T* start, const T* end, const T& value) noexcept;

} // namespace array_utils

#endif // ARRAY_UTILS_HPP