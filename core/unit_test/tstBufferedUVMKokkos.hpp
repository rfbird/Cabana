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

class TestTag
{
};

// TODO: dry with tstAoSoA
//---------------------------------------------------------------------------//
// Check the data given a set of values in an aosoa.
template <class aosoa_type>
void checkDataMembers( aosoa_type aosoa, const float fval, const double dval,
                       const int ival, const int dim_1, const int dim_2,
                       const int dim_3, int copy_back = 1 )
{
    //auto mirror =
        //Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
        //Cabana::create_mirror_view_and_copy( Kokkos::CudaHostPinnedSpace(), aosoa );

    //if ( copy_back == 0 )
    //{
        //auto mirror = aosoa;
    //}

    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );

    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
        {
            for ( int j = 0; j < dim_2; ++j )
            {
                for ( int k = 0; k < dim_3; ++k )
                {
                    EXPECT_EQ( slice_0( idx, i, j, k ), fval * ( i + j + k ) );
                }
            }
        }

        // Member 1.
        EXPECT_EQ( slice_1( idx ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( slice_2( idx, i ), dval * i );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( slice_3( idx, i, j ), dval * ( i + j ) );
    }
}

template <class buf_t>
class Tagfunctor_op
{
  public:
    // TODO: this needs a constructor and to actually use the data
    KOKKOS_INLINE_FUNCTION void operator()( const TestTag &,
                                            buf_t buffered_aosoa, const int s,
                                            const int a ) const
    {
    }
    KOKKOS_INLINE_FUNCTION void operator()(
        // const TestTag &,
        buf_t buffered_aosoa, const int s, const int a ) const
    {
    }
};

/*
void testBufferedTag()
{
    std::cout << "Testing buffered tag" << std::endl;
    using DataTypes = Cabana::MemberTypes<float>;

    // TODO: right now I have to specify the vector length so I can ensure it's
    // the same to do a byte wise async copy
    const int vector_length = 32;
    using AoSoA_t = Cabana::AoSoA<DataTypes, Kokkos::HostSpace, vector_length>;

    // Cabana::simd_parallel_for( policy_1, func_1, "2d_test_1" );

    int num_data = 512;
    AoSoA_t aosoa( "test tag aosoa", num_data );

    const int buffer_count = 3;
    const int max_buffered_tuples = 32;

    using buf_t = Cabana::BufferedAoSoA<buffer_count, TEST_EXECSPACE, AoSoA_t>;
    buf_t buffered_aosoa_in( aosoa, max_buffered_tuples );

    Tagfunctor_op<buf_t> func_1;

    Cabana::buffered_parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE, TestTag>( 0, aosoa.size() ),
        buffered_aosoa_in, func_1, "test buffered for tag" );
}
*/

void testBufferedKokkosCudaUVM()
{
    //int num_data = 1024; // * 1024;   // 2kb
    //int num_data = 1024 * 1024;       // 200mb
    //int num_data = 1024 * 1024 * 32;  // 6.4GB
    const int num_data = 1024 * 1024 * 112;   // 20GB (20199768064)

    // We want this to match the target space so we can do a fast async
    // copy
    //const int vector_length = 1024*1024*32; // SoA
    //const int vector_length = 32; // TODO: make this 32 be the default for GPU

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    //using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          //double[dim_1], double[dim_1][dim_2]>;

    //using space = Kokkos::CudaHostPinnedSpace;
    using space = Kokkos::CudaUVMSpace;

    Kokkos::View<float*[dim_1][dim_2][dim_3], space> v1("v1", num_data);
    Kokkos::View<int*, space> v2("v2", num_data);
    Kokkos::View<double*[dim_1], space> v3("v3", num_data);
    Kokkos::View<double*[dim_1][dim_2], space> v4("v4", num_data);

    // Declare the AoSoA type.
    // CudaHostPinned Space does not change performance (in 40eb8b7dff265b7f2ca580494c0355352856278a)
    //using AoSoA_t = Cabana::AoSoA<DataTypes, Kokkos::CudaHostPinnedSpace, vector_length>;
    //using AoSoA_t = Cabana::AoSoA<DataTypes, Kokkos::CudaUVMSpace, vector_length>;
    printf("Done creating views \n");
    //std::string label = "sample_aosoa";
    //AoSoA_t uvm_aosoa( label, num_data );

    //std::cout << "Making UVM array of size " << num_data << std::endl;
    //std::cout << num_data * sizeof(DataTypes) << " bytes " << std::endl;
    // Reset values so the outcome differs
    float fval = 4.4;
    double dval = 2.23;
    int ival = 2;

    //const auto slice_0 = Cabana::slice<0>(uvm_aosoa);
    //const auto slice_1 = Cabana::slice<1>(uvm_aosoa);
    //const auto slice_2 = Cabana::slice<2>(uvm_aosoa);
    //const auto slice_3 = Cabana::slice<3>(uvm_aosoa);

    printf("%d Starting parallel for ... \n", __LINE__);
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    auto f = KOKKOS_LAMBDA( const int z ) {
        // We have to call access and slice in the loop

        // We have to be really careful about how this access is
        // captured in the loop on GPU, and follow how ScatterView does
        // it safely. The `buffered_aosoa` may get captured by
        // reference, and then not be valid in a GPU context
        // auto buffered_access = buffered_aosoa.access();
        // auto buffered_access = buffered_aosoa.access();

        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
        {
            for ( int j = 0; j < dim_2; ++j )
            {
                for ( int k = 0; k < dim_3; ++k )
                {
                    v1( z, i, j, k ) = fval * ( i + j + k + sqrtf(i*j) );
                }
            }
        }

        // Member 1.
        v2(z ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
        {
            v3( z, i ) = dval * i+ sqrtf(i*dval+1) ;
        }

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
        {
            for ( int j = 0; j < dim_2; ++j )
            {
                v4( z, i, j ) = dval * ( i + j ) * sqrtf(dval-j);
            }
        }
    };

    //auto policy = Cabana::SimdPolicy< AoSoA_t::vector_length,
         //Kokkos::DefaultExecutionSpace >( 0, uvm_aosoa.size() );


    //Cabana::simd_parallel_for( policy,
        //f, "test_simd");
//
    Kokkos::parallel_for( num_data, f, "uvm kokoks");

    Kokkos::fence();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_taken = duration_cast<duration<double>>( t2 - t1 );

    printf("%d End parallel for. %e \n", __LINE__, time_taken.count() );

    // Reset the values on the host
    //fval = 4.4;
    //dval = 2.23;
    //ival = 2;

    high_resolution_clock::time_point t3 = high_resolution_clock::now();

    //auto policy2 = Cabana::SimdPolicy< AoSoA_t::vector_length,
         //Kokkos::DefaultHostExecutionSpace >( 0, uvm_aosoa.size() );

    //Cabana::simd_parallel_for( policy2,
        //f, "test_simd");

    Kokkos::fence();

    high_resolution_clock::time_point t4 = high_resolution_clock::now();
    time_taken = duration_cast<duration<double>>( t4 - t3 );

    //checkDataMembers( uvm_aosoa, fval, dval, ival, dim_1, dim_2, dim_3, 0 );
    //checkDataMembers( h_aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );
}


//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, bufferedUVMKokkosData_test ) { testBufferedKokkosCudaUVM(); }
//TEST( TEST_CATEGORY, cross_exec_test ) { testCrossExec(); }
// TEST( TEST_CATEGORY, bufferedData_tag_test ) { testBufferedTag(); }

} // namespace Test
