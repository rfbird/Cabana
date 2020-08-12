/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_BUFFEREDFOR_HPP
#define CABANA_BUFFEREDFOR_HPP

#include <Cabana_Parallel.hpp> // simd_for and SimdPolicy
#include <Kokkos_Core.hpp>

#include <cassert>

namespace Cabana
{

namespace
{
template <class ExecSpace>
struct SpaceInstance
{
    static ExecSpace create() { return ExecSpace(); }
    static void destroy( ExecSpace & ) {}
    static bool overlap() { return false; }
};

#ifndef KOKKOS_ENABLE_DEBUG
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct SpaceInstance<Kokkos::Cuda>
{
    static Kokkos::Cuda create()
    {
        cudaStream_t stream;
        cudaStreamCreate( &stream );
        return Kokkos::Cuda( stream );
    }
    static void destroy( Kokkos::Cuda &space )
    {
        cudaStream_t stream = space.cuda_stream();
        cudaStreamDestroy( stream );
    }
    static bool overlap()
    {
        bool value = true;
        auto local_rank_str = std::getenv( "CUDA_LAUNCH_BLOCKING" );
        if ( local_rank_str )
        {
            value = ( std::atoi( local_rank_str ) == 0 );
        }
        return value;
    }
};
#endif
#endif
} // namespace

template <class FunctorType, class extra_functor_arg_t, int VectorLength,
          class... ExecParameters>
inline void custom_simd_parallel_for(
    const SimdPolicy<VectorLength, ExecParameters...> &exec_policy,
    const FunctorType &functor, const extra_functor_arg_t &f_arg,
    const std::string &str = ""
    //,
    // typename std::enable_if<!std::is_same<typename SimdPolicy<VectorLength,
    // ExecParameters...>::work_tag, void>::value>::type = 0
)
{
    using simd_policy = SimdPolicy<VectorLength, ExecParameters...>;

    using work_tag = typename simd_policy::work_tag;
    using team_policy = typename simd_policy::base_type;

    using index_type = typename team_policy::index_type;

    // std::cout << "regular if" << std::endl;

    using ex = typename simd_policy::execution_space;

    /*
    auto f = KOKKOS_LAMBDA( extra_functor_arg_t buffered_aosoa,
            //std::forward<IndexTypes>( indices )...
            int s, int i )
    {
        functor(
                // t,
                buffered_aosoa, s, i
                // std::forward<IndexTypes>(indices)...
               );
    };
    */

    // TODO: this casues a seg fawult even if we don't use it???
    // Kokkos::DefaultExecutionSpace space1 =
    // SpaceInstance<Kokkos::DefaultExecutionSpace>::create(); std::cout <<
    // "Enabling async .. " << SpaceInstance<ex>::overlap() << std::endl;
    // SpaceInstance<ex>::destroy(space1);

    Kokkos::parallel_for(
        str, dynamic_cast<const team_policy &>( exec_policy ),
        KOKKOS_LAMBDA( const typename team_policy::member_type &team ) {
            index_type s = team.league_rank() + exec_policy.structBegin();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team, exec_policy.arrayBegin( s ),
                                           exec_policy.arrayEnd( s ) ),
                [&]( const index_type a ) {
                    Impl::functorTagDispatch<work_tag>( functor, f_arg, s, a );
                    // functor( f_arg, s, a);
                } );
        } );
}

/*
template <
class FunctorType, class extra_functor_arg_t, int VectorLength,
          class... ExecParameters
          >
inline void custom_simd_parallel_for(
    const SimdPolicy<VectorLength, ExecParameters...> &exec_policy,
    const FunctorType &functor, const extra_functor_arg_t &f_arg,
    const std::string &str = "",
          typename std::enable_if<std::is_same<typename SimdPolicy<VectorLength,
ExecParameters...>::work_tag, void>::value>::type = 0
    )
{
    using simd_policy = SimdPolicy<VectorLength, ExecParameters...>;

    using work_tag = typename simd_policy::work_tag;
    using team_policy = typename simd_policy::base_type;

    using index_type = typename team_policy::index_type;

    std::cout << "eanble if" << std::endl;

    // TODO: wrap functor to refresh capture of buffered aosoa, like
    // build_functor() used to

    Kokkos::parallel_for(
        str, dynamic_cast<const team_policy &>( exec_policy ),
        KOKKOS_LAMBDA( const typename team_policy::member_type &team ) {
            index_type s = team.league_rank() + exec_policy.structBegin();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team, exec_policy.arrayBegin( s ),
                                           exec_policy.arrayEnd( s ) ),
                [&]( const index_type a ) {
                    Impl::functorTagDispatch<work_tag>( functor, f_arg, s, a );
                    // functor( f_arg, s, a);
                } );
        } );
}
*/

// Requirements:
// 1) This must be user callable and seamlessly handle the data buffering
// 2) It must be able to accept 2D (simd) and 1D (Kokkos) loops/ range policies
// a) Safely peel/remainder loops if needed for the 2D case?
// 3) It should be able to warn in debug if you do something that we can detect
// as affecting performance, eg: a) warn if vector length does not make sense
// for target execution space
template <class BufferedAoSoA_t, class FunctorType, class... ExecParameters>
inline void buffered_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...> &exec_policy,
    BufferedAoSoA_t &buffered_aosoa,
    const FunctorType &functor, const std::string &str = "" )
{
    // TODO: passing a kokkos range policy and then building a simd policy
    // doesn't make a whole lot of sense?

    constexpr int VectorLength =
        BufferedAoSoA_t::from_AoSoA_type::vector_length;
    using simd_policy = SimdPolicy<VectorLength, ExecParameters...>;
    using work_tag = typename simd_policy::work_tag;
    // using team_policy = typename simd_policy::base_type;

    // using index_type = typename team_policy::index_type;

    int global_begin = exec_policy.begin();
    int global_end = exec_policy.end();
    int nelem = global_end - global_begin;

    // not ready for complex situations yet..
    assert( global_begin == 0 );
    assert( (size_t)global_end == buffered_aosoa.original_view.size() );

    // Calculate number buffer iterations needed to fit the data size
    int buffer_size = buffered_aosoa.buffer_size;
    int niter = nelem / buffer_size;

    std::cout << "running for " << niter << " buffered iterations "
              << std::endl;

    int begin = 0;
    int end = begin + buffer_size;

    // Load the first buffer, and block
    buffered_aosoa.load_buffer( 0 );
    Kokkos::fence();

    for ( int i = 0; i < niter; i++ )
    {
        // Now, for each iteration of the loop, we can:
        // 1) Run the code
        // 2) Push the next one down (this could be re ordered to be earlier?)
        // 3) Pull the last copy back

        std::cout << "Looping from " << begin << " to " << end << " which is "
                  << i * buffer_size << " in global space " << std::endl;

        simd_policy policy( begin, end );

        buffered_aosoa.slice_buffer(i);

        custom_simd_parallel_for( policy, functor, buffered_aosoa, str );
        // Cabana::simd_parallel_for( policy, functor, str );
        Kokkos::fence();

         auto s = Cabana::slice<0>(buffered_aosoa.internal_buffers[0]);
        std::cout << "buffer 0 at i=0 0 0 1 = " << s(0, 0, 0, 1) << " = " << s.access(0, 0, 0, 0, 1) << " pointer is " << &s(0,0,0,1) << std::endl;
        std::cout << "buffer 0 at i=0 0 0 2 = " << s(0, 0, 0, 2) << " = " << s.access(0, 0, 0, 0, 2) << " pointer is " << &s(0,0,0,2) << std::endl;
        if ( i < niter - 1 )
        { // Don't copy "next" on the last iteration
            buffered_aosoa.load_buffer( i + 1 );
        }

        Kokkos::fence();

        // copy all data back from localbuffer into the correct location in
        // global
        std::cout << "copy back buffer " << i % buffered_aosoa.get_buffer_count() << std::endl;

        buffered_aosoa.copy_buffer_back( i % buffered_aosoa.get_buffer_count(),
                                         buffer_size * ( i ) );
    }

    auto aosoa = buffered_aosoa.original_view;
    auto slice_0 = Cabana::slice<0>( aosoa );
    for (int idx = 0; idx < aosoa.size(); idx++)
    {
        int dim_1 = 3;
        int dim_2 = 2;
        int dim_3 = 4;
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    printf("aosoa slice 0 at %i %i %i %i = %e \n", idx, i, j, k,
                            slice_0( idx, i, j, k )
                //slice_0( idx )
              );
    }
}
} // namespace Cabana
#endif // CABANA_BUFFEREDFOR_HPP
