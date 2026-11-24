// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

#include "single_op_tests/range.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::RangeLayerTest;

namespace {

const std::vector<float> start = { 1.0f, 1.2f };
const std::vector<float> stop = { 5.0f, 5.2f };
const std::vector<float> step = { 1.0f, 0.1f };

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16
};

INSTANTIATE_TEST_SUITE_P(smoke_Basic, RangeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(start),
                                ::testing::ValuesIn(stop),
                                ::testing::ValuesIn(step),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        RangeLayerTest::getTestCaseName);
// Use runtime parameters to exercise the CPU kernel rather than constant folding.
TEST(smoke_RangeIntegerOverflow, ExtremeI32Inputs) {
    constexpr auto min = std::numeric_limits<int32_t>::min();
    constexpr auto max = std::numeric_limits<int32_t>::max();
    const auto start_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    const auto stop_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    const auto step_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    const auto range = std::make_shared<ov::op::v4::Range>(start_param, stop_param, step_param, ov::element::i32);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{range},
                                                  ov::ParameterVector{start_param, stop_param, step_param});
    ov::Core core;
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_CPU);
    auto request = compiled.create_infer_request();

    struct Case {
        std::array<int32_t, 3> inputs;
        std::vector<int32_t> expected;
    };
    const std::vector<Case> cases = {
        {{1, 0, min}, {1}},
        {{min, max, max}, {min, -1, max - 1}},
        {{0, 2, max}, {0}},
        {{max - 1, max, 2}, {max - 1}},
        {{max, min, -max}, {max, 0, -max}},
        {{0, min, min}, {0}},
    };
    for (const auto& test : cases) {
        SCOPED_TRACE(::testing::PrintToString(test.inputs));
        for (size_t i = 0; i < test.inputs.size(); ++i) {
            ov::Tensor input(ov::element::i32, ov::Shape{});
            input.data<int32_t>()[0] = test.inputs[i];
            request.set_input_tensor(i, input);
        }
        ASSERT_NO_THROW(request.infer());
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_shape(), ov::Shape{test.expected.size()});
        ASSERT_EQ(output.get_element_type(), ov::element::i32);
        const auto* data = output.data<const int32_t>();
        EXPECT_EQ(std::vector<int32_t>(data, data + output.get_size()), test.expected);
    }
}
}  // namespace
