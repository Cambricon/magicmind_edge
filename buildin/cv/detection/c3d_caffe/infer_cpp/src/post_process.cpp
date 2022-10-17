#include "post_process.hpp"
#include "utils.hpp"

std::vector<int> sort_indexes(const std::vector<float> &v, bool reverse)
{
    std::vector<int> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;
    if (reverse)
    {
        std::sort(idx.begin(), idx.end(),
                  [&v](int i1, int i2)
                  { return v[i1] > v[i2]; });
    }
    else
    {
        std::sort(idx.begin(), idx.end(),
                  [&v](int i1, int i2)
                  { return v[i1] < v[i2]; });
    }

    return idx;
}

