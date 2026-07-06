/**
  ******************************************************************************
  * @file    network_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-07-05T21:59:27+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "network_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_network_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_network_weights_array_u64[368] = {
  0x1b04c5fa1a81ca50U, 0xd1814ed2f71805d6U, 0x54cba0b3f5b9ab78U, 0x8106678726517f20U,
  0xb6f17fc844eb54cfU, 0x1741d59b55f0b45dU, 0x51732f35813f21aaU, 0x7a522e4ff3fbd381U,
  0xa3812329b72ed724U, 0x84900001186U, 0x21bfffff8feU, 0xfffffee800000092U,
  0x155ffffd75eU, 0xd1ed1a4b1fdf58fcU, 0x2ee0020b20f0feebU, 0xe3c9dfb2bb81e4ebU,
  0xcae01c1af2e81d05U, 0x32eff4ee12de0216U, 0xdef1cdb2b8baeefbU, 0xdef7e7faf0ea24f7U,
  0xea110ddaf719df05U, 0xf006cc81c7c2dc0fU, 0xbfbad23f35c90ee3U, 0x8be1128f3ce21e5U,
  0xbfc7cbf3c181eff2U, 0x8cebecdc17e7e9dfU, 0x10c81f330af21bf3U, 0xa9d2f7f7e18de3f9U,
  0xe2d90f21388f33cfU, 0x43c5423316e546f7U, 0xddea0223d28821f6U, 0xdd72cafee39a8f4U,
  0x12c90719410803e9U, 0xb49ef7d8fef4c1e4U, 0x100a11b72fef0207U, 0x73be0e332c0c1705U,
  0xb5e7e2e8fcad10dcU, 0x2914159ef928bd18U, 0x29e3022f7f001b08U, 0xd1b6eee034d604f3U,
  0xc13f541481e1daa2U, 0x171de3dcbcebf912U, 0x7a2a3200e92d0df4U, 0xb52804faabe7fecaU,
  0x171ac6f3d887effcU, 0x3c28ea0efdc12650U, 0xef5e0000de3407feU, 0xf61cfa18fc0a0110U,
  0x5a2ecd345a2f5a60U, 0x200b0b1a15d713ecU, 0xe8e3fb0ebde5d8f6U, 0xa4efd3b39b810d0dU,
  0x2bf7e5101b15060aU, 0x3d911e3c3c6dcebU, 0xc7e5eede9585f117U, 0x9fc01f917e2f508U,
  0xe7d100d9d0f0e10aU, 0xc4dbf1e98e90f424U, 0xf9f5e6f7351335efU, 0xfee2f2c4fae9e9dfU,
  0xe4dcdddbaeb6ee02U, 0x1fea221b230142f3U, 0x1909fdd20c8dce0aU, 0xdfe1cbd2b381f807U,
  0xff100e0211fe0deeU, 0x8eff08bf9f7cd10U, 0xbcdadfb6b8a1ea09U, 0xe51ffacbd6efcee7U,
  0xd0f3dfef10061be4U, 0x55e024a4b7f3beeU, 0xeb0dfef9cfbae3f6U, 0xfed8dce32bcc0a1cU,
  0x2d22322127f7031eU, 0xe71c00d2de05d1d2U, 0xeee9eb3024ed1921U, 0x3d4a07455e1524e8U,
  0x90ea2608e515d9a6U, 0xcef7f317c7ed1195U, 0xe0dbe0afb3819caeU, 0x9cec0fe7d00aeb8cU,
  0xbc150816e3fcfa93U, 0xd9f5e28bd5bcacdcU, 0xa40318d5ce18c6aeU, 0xd8d21811c6f8c495U,
  0xd3e9dfaad4be8ef1U, 0xd31bcaddd7bd03f0U, 0xe208f1e8f514262aU, 0xe5dc0b2d7f393033U,
  0xe208c8cdd6f9f0d2U, 0xc7f618e9d614f1d1U, 0x3f2631303c0ef730U, 0xe2f2e4cbcaf3c906U,
  0xe1120be7eb431220U, 0x40090ae66118f73fU, 0x9c24ffccf8073b16U, 0xdc47a1c535a7f02dU,
  0x200e6ef33e41e7fU, 0xd4fe26ebb3db16ddU, 0xf714bcdcde85031bU, 0x150ae8ac2ddc2631U,
  0xad3060d9c339d2f3U, 0x81e9abfd2e2d12fU, 0x5266bff210143d41U, 0x7bf015429d82effU,
  0xb392f51901b701f7U, 0xeba51500d781ea06U, 0x2aac0045440336e9U, 0xcab1020ff2fff4fdU,
  0xeacb151a10e7170cU, 0xf7c5231f21130ff8U, 0xd1df050908edfcfdU, 0xcec0fcffebd610efU,
  0xf9210787bec0a5c9U, 0x2de7db0748cefb16U, 0x1cfbbedd2feb28f8U, 0xfb1cef82e2c29fddU,
  0xfcf8d0ed12bff6fcU, 0x2ef5ce0d680e38f9U, 0xe33fefa6c306b3ceU, 0x34dbd00f7fcc1227U,
  0xff02f5206a430bf7U, 0x102c0482b39696c2U, 0xeaefd1fe66ec0edfU, 0x11eef6cc6e30e234U,
  0xd84deec9d001a9f3U, 0x2de3f134fa0a0c0bU, 0xe6e6ec0dfc0217d7U, 0xfc2103a3d018cbfcU,
  0xffdae0fc7feafee8U, 0x18e4d7292d16123aU, 0xfa39f8edf8e4d9ccU, 0xf206ea0e22d40a13U,
  0xff22e7fd3f4f370bU, 0xf5f628f9c604d7c6U, 0x29f0f9e901ffe406U, 0xfdf1e6144900e9f5U,
  0xda01fbcbe301d6f6U, 0x1418e9e4fff905e2U, 0x2013c70e7f0a3aeaU, 0xf21f5310a2e0fa04U,
  0x2c44da04ddbfd730U, 0x33361e07d51cec0fU, 0xbf0c00f2a1cbeadeU, 0x1644dffede81db0cU,
  0x3de4e916d4a4021fU, 0xd1184210de3c1501U, 0x1041c5ed1c190c46U, 0x5241f2290465313cU,
  0xfc3eecfe01fe3949U, 0xbe2505ea1a1bd9e1U, 0xab4b4322ef26ed16U, 0x1dedeb1303dff422U,
  0xb14f9f22af704beU, 0x44cfbe3dce512beU, 0x222bf60710421f1dU, 0xf1fc21fb183cefcbU,
  0xea7f07ffcb2a01bbU, 0x8370000094eU, 0xfffffa15ffffff14U, 0xd7b000010acU,
  0x14f0fffffff7U, 0x56f00000206U, 0x807fffffe0dU, 0x365000001d1U,
  0xfffff4f6fffff657U, 0xbea280aff3c35U, 0x1a5ad1f4de14061bU, 0x5fde9ee4320dd07U,
  0x6065e2f00fe34127U, 0xb12381ce4712ccb8U, 0x684450183c1a3f6bU, 0x181122f822b00b10U,
  0x2524d4e6fecf442bU, 0x15ea25c821dba617U, 0x1d00292025046510U, 0xd053ceaa1329dcd8U,
  0x7b5a22592beb6f5cU, 0xed0b0f230eec0d20U, 0x150fd30eeddb2404U, 0x170ae3fefd34f719U,
  0x18420c0417fc17f4U, 0xd528ccbc3fead9b5U, 0x6f360d362b0f1b38U, 0xf0081afd0ad2ec19U,
  0x1516e2f6e2cc0210U, 0x1815e804edfdc012U, 0x201a09f6f6e93f04U, 0xc324dccb2711bcdcU,
  0x7f1d311f2dfa562dU, 0x6f3d349d5d4041aU, 0x15e6b9fffafe3614U, 0xd0ae00c3d61e3f5U,
  0xf122dce912d037c5U, 0xad1c97f35fdff3aaU, 0x485c011d1f0e63d9U, 0x20013b4711b5e6d6U,
  0x38cad2dbf6d03c2eU, 0xe90d131deafbf011U, 0x110a0717cfbd7feeU, 0xf949e2bfe532e6d9U,
  0x7ce60b30241a4f40U, 0xa3261fa8d84222e8U, 0xcfec1d254c4af42aU, 0x1b04041ec91a5efdU,
  0xa6c4f416d519da14U, 0x2ee12048bdf41233U, 0xbde2e2cdb933cfc8U, 0xa3f6e89bd93e2dd7U,
  0xe6d6fe22272de4e9U, 0x270bc922a3294dffU, 0xc2c3f4f9d22ca3f3U, 0x30af3130a9fa044bU,
  0x81b3949ba9eb14b5U, 0x10f9fef709071121U, 0x161f12ebefe30f0fU, 0xdf521e52623fe0cU,
  0xf833e706e8f31de9U, 0xdb1ecad72dfc00bfU, 0x65220a1f28fe0d11U, 0x12fdfd25fbcee7faU,
  0xfc0ee9e5f1012b18U, 0xf2fbf8e40d13d82dU, 0x180b0b1ef0f55325U, 0xee36dded2f19fdd9U,
  0x7f06fe1427f93039U, 0xb1e6dfb1e241f3d5U, 0xeaec333e3252f0f7U, 0x2fd4f807ca225b1fU,
  0xf5efe5ddd734d5f2U, 0x47b6652fdc1f1217U, 0xaccab4de97fee1aeU, 0xbb29d5aadb2f04e4U,
  0x5b7243b5250d0feU, 0x10fa011ab1153ff5U, 0xceaacdecdb29bbf7U, 0x218f5e48bad80333U,
  0xc1a6bcc7a824da81U, 0xda2fb043b7b7a948U, 0xcd84d2e71dc0a222U, 0x9cc1e6db16b10720U,
  0xfbb4fe159f03d602U, 0xec1c1f343b897d3U, 0xa4b7d2599f95d58cU, 0xd148222ccfce9c46U,
  0x31f3972e0829f589U, 0xfb93c6f8bef3b7efU, 0x1ed281b042e0cac8U, 0x2bfa9cbf88d403ecU,
  0xd88a05edea2ed723U, 0xa5e3d2d1ac2931caU, 0xb02201d51be6eb2dU, 0xa0e4f6c7f5cefbdcU,
  0xe436e8f217fbed10U, 0x1f30cc3225e2a3e8U, 0xeba581f5fe1c00ffU, 0xfa8dfaeb99c4a01eU,
  0x3bbafc228d201739U, 0x1201a4020b1cb0f0U, 0xcffc97e296f1ca92U, 0xa7aece0bb702c1caU,
  0xbfdc990724a8b8beU, 0x16e3f639021b0af9U, 0x4131cedde1e02123U, 0xe9ebf7ef135cd638U,
  0x1743062316d148f5U, 0xc336b7a311d4d1b8U, 0x546cea323b2935e9U, 0xe9141735d5c309e6U,
  0x1b18dbfbf609143fU, 0xaf3efffd607dd59U, 0x8e631ee15114735U, 0xda02a6d132f5b4a1U,
  0x7ffc460a39123554U, 0xffebdf0427fafd12U, 0x1c40ddebf0ecf6f7U, 0x411fbd6fd0ef60aU,
  0x90afc2816ed01e9U, 0xdcf7c5e54e0daad5U, 0x7f3f29202fe0452cU, 0xe210f60bd3e2e301U,
  0xfe0b0b07d9dd3309U, 0x17e012cc06e2f01cU, 0x173de30c01f26bf8U, 0xeb28afb9f708f8d9U,
  0x7a2d47311d264507U, 0xa00302c6e745fd02U, 0xf9cd103634330115U, 0xfbe5d2fded0227edU,
  0xf3baf0cee21eb8f7U, 0x5481364bab00f84aU, 0x9ee2b6bbe3f3e79cU, 0xb924f6c7e63d09beU,
  0xe9d6113a191e0709U, 0xfcf0cde514104905U, 0xe7e5d417fe1b9dccU, 0x3ca75328affb273cU,
  0xbfdda1c0e5e6f2c2U, 0x1704e6f522fa0cf5U, 0x237ff06eff50f0cU, 0x1feaf9de1010ea1bU,
  0x1024060b1311f6e5U, 0xd52098df571ccfd2U, 0x3c26371232fa0b09U, 0x11ed242adfe5e835U,
  0x32fd05f508d90e02U, 0xf5271debe4e4f824U, 0x1e0b1a011ee163fdU, 0xea27dabc15fec4cbU,
  0x7f1f0f2efeee4a12U, 0xd9e9ef0ff21a101aU, 0x3a1d03d515fd17e2U, 0x180f03004658e421U,
  0x2040f903e9e821fbU, 0xd3199bbc1200cec4U, 0x7f4e322d40e34b2dU, 0x2bec0321dbd1c026U,
  0x15df0ee2d7db3af6U, 0xf5e21c1f1903ba27U, 0xfc2a02030cfe2e3fU, 0xf13adbdc21efb3c7U,
  0x5224433b142b3a2dU, 0xc8fa0099e743e3faU, 0xbc4033f4434f317U, 0x12f1d40ebef04affU,
  0x7d4e901f449dbe8U, 0x4b813e58a6103b49U, 0xbce6bfd1b610e195U, 0xacff03c60b40eeacU,
  0xab4121f472ee3dfU, 0xf7100b00dd0930f3U, 0xeab1ca04ef34ccf7U, 0x45d22f41a1fc0825U,
  0xebd6d09d93fae5a7U, 0x2a0c112a0df81c08U, 0x302702020acefaffU, 0x220d29f51b2dc803U,
  0xf813f60917f13efaU, 0xc94da2cc4e01d2c1U, 0x3647282062083552U, 0xf9d92ffb17e0d736U,
  0x27ebd3cdf7c7142eU, 0x1cfc26eef9dbe647U, 0x2f210f0917fd321dU, 0xc541b0ce36e4e0c5U,
  0x7ff3fc30fb264e0fU, 0xf50d2b0c013b08U, 0xf421e7e9e2e10813U, 0x2ad704c21550e200U,
  0x1d5620f0fcd901e6U, 0xd5f788ad45d3e1dcU, 0x7f412f0f51e21923U, 0x8dbe623f3c5d816U,
  0x3ee9fd1df2fe2afcU, 0x250fded1ee1db3feU, 0x45fc2b25e0fe3801U, 0xdb3cc09a2335bdc2U,
  0x5f45193b0ff5214eU, 0xfffffffafffffed2U, 0xe19000008d9U, 0xe3c0000024aU,
  0xfffff18cfffff4dcU, 0x3a000000c78U, 0x31000000cb1U, 0xd9a0000074eU,
  0xa950000055cU, 0x361f68d762cde9e6U, 0xbee87fd3d276d8b8U, 0xcdbU,
};


ai_handle g_network_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_network_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

