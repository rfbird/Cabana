#ifndef CABANA_DATA_REGISTRY_H
#define CABANA_DATA_REGISTRY_H

#include <unordered_map>

/**
 * @brief This class is a key value pair store that is meant to track the
 * registration, allocation, and deletion of Kokkos/Cabana views in order to
 * allow Fortran to retain access to different views without the need for one
 * global static per view.
 *
 * It is meant to facilitate the retrieval of objects, but also aid in making
 * the allocation and deletion more transparent from a Fortran context
 */
template <class K, class V> class data_registry
{
    // TODO: should we register the .data() element too?
    // TODO: we could even record the real underlying (AoSoA) type if available
    using underlying_store_t = std::unordered_map<K, V*>;
    underlying_store_t ds;

    public:
        // TODO: in the real version we will have to be careful with have we
        // access these maps



        // TODO: add could grab the underlying name from AoSoA?
        void add(K k, V* v)
        {
            ds[k] = v;
        }
        void remove(K k)
        {
            ds.erase(k);
        }

        void* get(K k)
        {
            return ds[k];
        }

        // TODO: this likely need a cast to call the data
        void* get_data(K k)
        {
            return ds[k];
        }

        double* get_slice_double(K k, size_t index)
        {
            return ds[k];
        }

        // Templated specialization of get that can do the casting for you
        template<typename AoSoA, typename index> typename AoSoA::template member_slice_type<index> get_data(K k)
        {
            auto& a = static_cast<AoSoA>(ds[k]);
            return Cabana::slice<index>(a);
        }
};

#endif // CABANA_DATA_REGISTRY_H
