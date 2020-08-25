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

#include <Cabana_AoSoA.hpp>
#include <Cabana_BufferedAoSoA.hpp>
#include <Cabana_BufferedFor.hpp>
#include <Cabana_Types.hpp>

#include <gtest/gtest.h>

namespace Test
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

// HOSTSPACE ONLY
//---------------------------------------------------------------------------//
// Check the data given a set of values in an aosoa.
template <class aosoa_t>
void checkDataMembers( aosoa_t &aosoa, const float fval, const double dval,
                       const int ival, const int dim_1, const int dim_2,
                       const int dim_3 )
{
    // auto mirror =
    // Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );

    auto f = KOKKOS_LAMBDA( const int idx )
    {
        int s = 0;
        int a = 0;

        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
        {
            for ( int j = 0; j < dim_2; ++j )
            {
                for ( int k = 0; k < dim_3; ++k )
                {
                    EXPECT_EQ( slice_0.access( s, a, i, j, k ),
                               fval * ( i + j + k ) );
                }
            }
        }

        // Member 1.
        EXPECT_EQ( slice_1.access( s, a ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( slice_2.access( s, a, i ), dval * i );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( slice_3.access( s, a, i, j ), dval * ( i + j ) );
    };

    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
    {
        f( idx );
    }

    // Cabana::SimdPolicy<aosoa_t::vector_length, Kokkos::HostSpace> policy_1(
    // 0, aosoa.size() );
    // Cabana::simd_parallel_for( policy_1, f, "check" );
    // Kokkos::parallel_for("check", Kokkos::RangePolicy<Kokkos::HostSpace>(0,
    // aosoa.size() ), f);
}

template <typename aosoa_t, typename exec_space_t>
void set_val( aosoa_t &aosoa, const float fval, const double dval,
              const int ival, const int dim_1, const int dim_2, const int dim_3,
              exec_space_t space, bool async )
{
    const auto slice_0 = Cabana::slice<0>( aosoa );
    const auto slice_1 = Cabana::slice<1>( aosoa );
    const auto slice_2 = Cabana::slice<2>( aosoa );
    const auto slice_3 = Cabana::slice<3>( aosoa );

    auto f = KOKKOS_LAMBDA( const int idx )
    {
        // Value update
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
        {
            for ( int j = 0; j < dim_2; ++j )
            {
                for ( int k = 0; k < dim_3; ++k )
                {
                    slice_0( idx, i, j, k ) = fval * ( i + j + k );
                }
            }
        }

        // Member 1.
        slice_1( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
        {
            slice_2( idx, i ) = dval * i;
        }

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
        {
            for ( int j = 0; j < dim_2; ++j )
            {
                slice_3( idx, i, j ) = dval * ( i + j );
            }
        }
    };

    // Cabana::SimdPolicy<aosoa_t::vector_length, exec_space> policy_1(
    // 0, aosoa.size() );
    // Cabana::simd_parallel_for( policy_1, f, "test set" );
    if ( async )
    {
        std::cout << "Enabling overlap? .. "
                  << SpaceInstance<exec_space_t>::overlap() << std::endl;

        Kokkos::parallel_for(
            "set1d",
            Kokkos::Experimental::require(
                Kokkos::RangePolicy<exec_space_t>( space, 0, aosoa.size() ),
                Kokkos::Experimental::WorkItemProperty::HintLightWeight ),
            f );
    }
    else
    {
        Kokkos::parallel_for(
            "set1d", Kokkos::RangePolicy<exec_space_t>( 0, aosoa.size() ), f );
    }
}

template <typename aosoa_t, typename exec_space = TEST_EXECSPACE>
void set_val_simd( aosoa_t &aosoa, const float fval, const double dval,
                   const int ival, const int dim_1, const int dim_2,
                   const int dim_3 )
{
    const auto slice_0 = Cabana::slice<0>( aosoa );
    const auto slice_1 = Cabana::slice<1>( aosoa );
    const auto slice_2 = Cabana::slice<2>( aosoa );
    const auto slice_3 = Cabana::slice<3>( aosoa );

    auto f = KOKKOS_LAMBDA( const int s, const int a )
    {
        // Value update
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
        {
            for ( int j = 0; j < dim_2; ++j )
            {
                for ( int k = 0; k < dim_3; ++k )
                {
                    slice_0.access( s, a, i, j, k ) = fval * ( i + j + k );
                }
            }
        }

        // Member 1.
        slice_1.access( s, a ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
        {
            slice_2.access( s, a, i ) = dval * i;
        }

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
        {
            for ( int j = 0; j < dim_2; ++j )
            {
                slice_3.access( s, a, i, j ) = dval * ( i + j );
            }
        }
    };

    Cabana::SimdPolicy<aosoa_t::vector_length, exec_space> policy_1(
        0, aosoa.size() );
    Cabana::simd_parallel_for( policy_1, f, "test set" );
}

void tstOverlap( bool async = true )
{
    // We want this to match the target space so we can do a fast async
    // copy
    const int vector_length = 32; // TODO: make this 32 be the default for GPU

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_EXECSPACE, vector_length>;

    // using AoSoA_h_t = Cabana::AoSoA<DataTypes,
    // Kokkos::DefaultHostExecutionSpace, vector_length>;
    using AoSoA_h_t =
        Cabana::AoSoA<DataTypes, Kokkos::HostSpace, vector_length>;

    int num_data = 1024 * 256;
    // int num_data = 64;

    AoSoA_t aosoa1( "aosoa 1", num_data );
    AoSoA_t aosoa2( "aosoa 2", num_data );
    AoSoA_t aosoa3( "aosoa 3", num_data );

    float init_fval = 3.4;
    double init_dval = 1.23;
    int init_ival = 1;

    // Init
    set_val_simd( aosoa1, init_fval, init_dval, init_ival, dim_1, dim_2,
                  dim_3 );
    set_val_simd( aosoa2, init_fval, init_dval, init_ival, dim_1, dim_2,
                  dim_3 );
    set_val_simd( aosoa3, init_fval, init_dval, init_ival, dim_1, dim_2,
                  dim_3 );

    // If you use mirror view here you don't get the exact right layout
    AoSoA_h_t h1( "host aosoa 1", num_data );
    AoSoA_h_t h2( "host aosoa 2", num_data );
    AoSoA_h_t h3( "host aosoa 3", num_data );

    // Fill host copy with inited data
    Cabana::deep_copy( h1, aosoa1, async );
    Cabana::deep_copy( h2, aosoa2, async );
    Cabana::deep_copy( h3, aosoa3, async );

    /////// START ///////
    // copy to device
    Cabana::deep_copy( aosoa1, h1, async );
    Cabana::deep_copy( aosoa2, h2, async );
    Cabana::deep_copy( aosoa3, h3, async );

    float fval = 4.4;
    double dval = 2.23;
    int ival = 2;

    // Change Val
    set_val_simd( aosoa1, fval, dval, ival, dim_1, dim_2, dim_3 );
    set_val_simd( aosoa2, fval, dval, ival, dim_1, dim_2, dim_3 );
    set_val_simd( aosoa3, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Copy back
    Cabana::deep_copy( h1, aosoa1, async );
    Cabana::deep_copy( h2, aosoa2, async );
    Cabana::deep_copy( h3, aosoa3, async );

    Kokkos::fence();
    // check
    checkDataMembers( h1, fval, dval, ival, dim_1, dim_2, dim_3 );
}

void tstOverlapingSpaces( bool async = true )
{
    // We want this to match the target space so we can do a fast async
    // copy
    const int vector_length = 32; // TODO: make this 32 be the default for GPU

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    TEST_EXECSPACE target_async_space1 =
        SpaceInstance<TEST_EXECSPACE>::create();
    TEST_EXECSPACE target_async_space2 =
        SpaceInstance<TEST_EXECSPACE>::create();

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_EXECSPACE, vector_length>;

    // using AoSoA_h_t = Cabana::AoSoA<DataTypes,
    // Kokkos::DefaultHostExecutionSpace, vector_length>;
    using AoSoA_h_t =
        Cabana::AoSoA<DataTypes, Kokkos::HostSpace, vector_length>;

    int num_data = 1024;
    // int num_data = 64;

    AoSoA_t aosoa1( "aosoa 1", num_data );
    AoSoA_t aosoa2( "aosoa 2", num_data );
    // AoSoA_t aosoa3( "aosoa 3", num_data );

    float init_fval = 3.4;
    double init_dval = 1.23;
    int init_ival = 1;

    using namespace std::chrono;
    // Init
    set_val( aosoa1, init_fval, init_dval, init_ival, dim_1, dim_2, dim_3,
             target_async_space1, async );
    set_val( aosoa2, init_fval, init_dval, init_ival, dim_1, dim_2, dim_3,
             target_async_space2, async );
    // set_val(aosoa3, init_fval, init_dval, init_ival, dim_1, dim_2, dim_3,
    // target_async_space, async);

    // If you use mirror view here you don't get the exact right layout
    AoSoA_h_t h1( "host aosoa 1", num_data );
    AoSoA_h_t h2( "host aosoa 2", num_data );
    // AoSoA_h_t h3( "host aosoa 3", num_data );

    // Fill host copy with inited data
    Cabana::deep_copy( h1, aosoa1, async );
    Cabana::deep_copy( h2, aosoa2, async );
    // Cabana::deep_copy( h3, aosoa3, async );

    /////// START ///////
    // copy to device
    Cabana::deep_copy( aosoa1, h1, async );
    Cabana::deep_copy( aosoa2, h2, async );
    // Cabana::deep_copy( aosoa3, h3, async );

    Kokkos::fence(); // fence before async timing

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    float fval = 4.4;
    double dval = 2.23;
    int ival = 2;

    // Change Val
    set_val( aosoa1, fval, dval, ival, dim_1, dim_2, dim_3, target_async_space1,
             async );
    set_val( aosoa2, fval, dval, ival, dim_1, dim_2, dim_3, target_async_space2,
             async );
    // set_val(aosoa3, fval, dval, ival, dim_1, dim_2, dim_3,
    // target_async_space, async);

    Kokkos::fence(); // fence after timing
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>( t2 - t1 );

    std::cout << "It took " << time_span.count() << " seconds." << std::endl;

    // Copy back
    Cabana::deep_copy( h1, aosoa1, async );
    Cabana::deep_copy( h2, aosoa2, async );
    // Cabana::deep_copy( h3, aosoa3, async );

    // check, on host
    checkDataMembers( h1, fval, dval, ival, dim_1, dim_2, dim_3 );

    SpaceInstance<TEST_EXECSPACE>::destroy( target_async_space1 );
    SpaceInstance<TEST_EXECSPACE>::destroy( target_async_space2 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, bufferedData_test_async ) { tstOverlap( true ); }
TEST( TEST_CATEGORY, bufferedData_test_noasync ) { tstOverlap( false ); }
TEST( TEST_CATEGORY, bufferedData_test_spaces_async )
{
    tstOverlapingSpaces( true );
}
TEST( TEST_CATEGORY, bufferedData_test_spaces_noasync )
{
    tstOverlapingSpaces( false );
}

} // namespace Test
