//----------舍入部分修改-------------------------------------------------------
// 执行舍入规则，如果第 24 位是 1，则需要进位。如果第 24 位是 0，则直接截断。
        if (significand_32_update[24])
        {
            int carry = 1;
            for (int i = 24; i >= 0 && carry; i--)
            {
                significand_32_update[i] += carry;
                if (significand_32_update[i] > 1)
                {
                    significand_32_update[i] = 0;
                    carry = 1;
                }
                else
                {
                    carry = 0;
                }
            }
        }
//------------------------------------------------------------------------------

struct TensorQ *test_tensor = getTensorQ(4);
    test_tensor->lens[0] = 1;
    test_tensor->lens[1] = 256;
    test_tensor->lens[2] = 40;
    test_tensor->lens[3] = 40;
    int test_len =  getlengthQ(test_tensor);
    char *ttdata = (char *)readfile("output_result_file/_model.19_cv2_act_Mul_output_0_quantized");
    test_tensor->data = (int *) malloc(sizeof(int) * test_len);
    for(int i=0; i<test_len; i++){
        test_tensor->data[i] = (int)ttdata[i];
    }
    
    output_gen_x_no_transpose(test_tensor, "test_tensor");

    QLinearConv_AUTO(
        instruction_out, 0x0, 0x10000000,
        "TEST",
        test_tensor,
        "initializer/_model.19_cv2_act_Mul_output_0_scale", "initializer/_model.19_cv2_act_Mul_output_0_zero_point",
        "initializer/model.20.cv1.conv.weight_quantized",
        "initializer/model.20.cv1.conv.weight_scale", "initializer/model.20.cv1.conv.weight_zero_point",
        "initializer/_model.20_cv1_conv_Conv_output_0_scale", "initializer/_model.20_cv1_conv_Conv_output_0_zero_point",
        "initializer/model.20.cv1.conv.bias_quantized",
        TEST_OUTPUT,
        1,1,0,1
    )
    result_compare(1,256,40,40,TEST_OUTPUT, "output_result_file/_model.20_cv1_conv_Conv_output_0_quantized");
    output_gen_x_no_transpose(TEST_OUTPUT, "output");
    return 0;







