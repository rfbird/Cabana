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

template <typename aosoa_t >
void set_val( aosoa_t &aosoa, const float fval, const double dval,
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

    Cabana::SimdPolicy<aosoa_t::vector_length, TEST_EXECSPACE> policy_1(
        0, aosoa.size() );
    Cabana::simd_parallel_for( policy_1, f, "test set" );
}

void tstPartialDeepCopy()
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

    int num_data = 1024;
    // int num_data = 64;

    AoSoA_t   d1( "device 1", num_data );
    AoSoA_t   d2( "device 2", num_data );
    AoSoA_h_t h1( "host 1", num_data*2 );

    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> kd1("kd1", num_data);
    Kokkos::View<float*, Kokkos::DefaultHostExecutionSpace> kh1("kh1", num_data);

    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;

    // Init
    set_val( d1, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Try copy Cabana objects of different size => should fail?
    //Cabana::deep_copy( h1, d1 );

    // Try copy Kokkos Views of different size => should fail?
    Kokkos::deep_copy( kh1, kd1 );
    Kokkos::deep_copy( kd1, kh1 );

    /*
    // Fill host copy with inited data
    Cabana::deep_copy( h1, aosoa1 );

    /////// START ///////
    // copy to device
    Cabana::deep_copy( aosoa1, h1 );

    float fval = 4.4;
    double dval = 2.23;
    int ival = 2;

    // Change Val
    set_val( aosoa1, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Copy back
    Cabana::deep_copy( h1, aosoa1 );
    */

    // check, on host
    //checkDataMembers( h1, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( TEST_CATEGORY, partial_deep_copy ) { tstPartialDeepCopy(); }

} // namespace Test
