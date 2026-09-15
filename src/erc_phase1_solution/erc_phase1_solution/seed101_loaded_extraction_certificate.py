"""Exact offline certificate for the paused seed-101 loaded recovery.

This data is inert: importing the module performs no ROS, controller, Gazebo,
filesystem, or network operation.  The certificate is valid only for the
recorded checkpoint.  Its two commanded phases are five 1 mm world-up
micro-lifts followed by forty-nine 5 mm fixed-orientation world-minus-X legs.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_ORIGINAL_JSON_FILE_SHA256 = (
    'e400ee79e5f7e3e5fe826e9a305837c0fd19daf14603d385274296bd665e3ed7'
)
CERTIFICATE_SOURCE_JSON_SHA256 = (
    '7a4b661bda14b29431587e5795a162a83a2d4b46290a523f22e10753ad732374'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    'c8dd9885be723dc7e1c795cbc8098f001851a0e021162fd811f453566ebf3e25'
)
CERTIFICATE_JSON = r'''{
  "attached_book_corners_padding_15mm_in_grasp": [
    [
      -0.07620839574197066,
      -0.03007083372950118,
      0.108736626864067
    ],
    [
      0.05812465957906762,
      -0.030075944775654107,
      -0.1369351668404446
    ],
    [
      0.09048911905726388,
      -0.03197004818065538,
      0.19988667647583364
    ],
    [
      0.2248221743783022,
      -0.03197515922680831,
      -0.04578511722867799
    ],
    [
      -0.0756816484190892,
      0.029926168665790315,
      0.1090234034970464
    ],
    [
      0.05865140690194909,
      0.029921057619637384,
      -0.13664839020746522
    ],
    [
      0.09101586638014537,
      0.028026954214636114,
      0.20017345310881302
    ],
    [
      0.22534892170118365,
      0.028021843168483183,
      -0.045498340595698614
    ]
  ],
  "attached_book_corners_physical_in_grasp": [
    [
      0.06448363650904519,
      0.014772142843872559,
      -0.11636303886895669
    ],
    [
      0.2048604910768217,
      0.013172804358690077,
      -0.03960510235378477
    ],
    [
      0.06422026284760435,
      -0.015226358353773173,
      -0.11650642718544618
    ],
    [
      0.20459711741538059,
      -0.016825696838955656,
      -0.03974849067027467
    ],
    [
      -0.055456591456167575,
      0.014776706277937661,
      0.10298677693864305
    ],
    [
      0.08492026311160866,
      0.01317736779275518,
      0.17974471345381457
    ],
    [
      -0.0557199651176087,
      -0.015221794919708069,
      0.10284338862215314
    ],
    [
      0.08465688945016782,
      -0.016821133404890554,
      0.17960132513732507
    ]
  ],
  "audit": {
    "arm_and_closed_gripper_shelf_collision_tolerance_m": 0.00075,
    "collision_free": true,
    "dense_samples": 394,
    "expected_support_floor_contact_only_until_lift": true,
    "final_book_shelf_face_clearance_m": 0.026355635550713075,
    "maximum_fk_orientation_error_rad": 8.633568591784384e-06,
    "maximum_fk_position_error_m": 9.472530517954066e-06,
    "maximum_joint_increment_rad": 0.002,
    "minimum_15mm_padded_payload_robot_aabb_clearance_m": 0.006788639611009117,
    "minimum_book_back_clearance_m": 0.08135526095131818,
    "minimum_book_ceiling_clearance_m": 0.044983883843970895,
    "minimum_carried_book_other_book_aabb_clearance_m": 0.27618833797572817,
    "minimum_closed_gripper_shelf_triangle_aabb_clearance_m": 0.011025032549784708,
    "minimum_robot_bin_aabb_clearance_m": 2.3314081168205667,
    "minimum_robot_other_book_aabb_clearance_m": 0.13924579006711044,
    "minimum_robot_table_aabb_clearance_m": 2.173487401159301,
    "minimum_self_aabb_clearance_m": 0.0020226386064607915
  },
  "execution_policy": {
    "keep_current_gripper_pressure": true,
    "maximum_relative_rotation_slip_per_leg_rad": 0.002,
    "maximum_relative_translation_slip_per_leg_m": 0.0005,
    "micro_lift_count": 5,
    "micro_lift_step_m": 0.001,
    "minimum_micro_lift_duration_s": 0.8,
    "minimum_outward_duration_s": 1.0,
    "outward_count": 49,
    "outward_step_m": 0.005,
    "require_fresh_bilateral_contact_each_leg": true
  },
  "input": {
    "base_world_xyyaw": [
      2.0027012233,
      -0.1494027021,
      -0.7853981633974483
    ],
    "book_physical_bounds_world_m": [
      [
        2.8133231333478568,
        -0.1689852271047326,
        1.4518548043981394
      ],
      [
        2.9736599134414936,
        -0.1371858235630853,
        1.701865190474155
      ]
    ],
    "book_position_world_m": [
      2.893491523394675,
      -0.15308552533390896,
      1.5768599974361472
    ],
    "book_quaternion_xyzw": [
      -0.003965224583946054,
      0.7071136498551619,
      0.003912055206789831,
      0.7070779723669585
    ],
    "gripper_master_m": 0.02900822,
    "head_q2_measured": [
      -5.341047622287229e-12,
      -2.104500678413703e-12
    ],
    "left_q8_measured": [
      0.349999786476243,
      -0.3110564645991377,
      0.6140707236141174,
      0.08158577121790486,
      -1.4806815491355565,
      0.17284131144392148,
      1.0327390973509902,
      0.21572246882942994
    ],
    "right_q7_measured": [
      1.1017502906726425e-06,
      2.930440907271199e-06,
      9.312227186970225e-07,
      -7.4406713585614e-06,
      2.550846057202886e-08,
      5.052158441872368e-10,
      -2.9108287519463966e-10
    ]
  },
  "kind": "read_only_partial_step2_loaded_recovery_certificate",
  "route": [
    {
      "expected_book_back_clearance_m": 0.0813556365585062,
      "expected_book_ceiling_clearance_m": 0.04999326952584493,
      "expected_book_center_world_m": [
        2.893491523394675,
        -0.15308552533390896,
        1.5768599974361472
      ],
      "expected_book_floor_signed_m": -3.66246549288185e-06,
      "expected_book_shelf_face_clearance_m": -0.21864436344149363,
      "expected_hand_world_position_m": [
        2.8128948235882874,
        -0.154202697623061,
        1.5688308112779592
      ],
      "lift_world_z_m": 0.0,
      "outward_world_minus_x_m": 0.0,
      "phase": "start",
      "q8": [
        0.349999786476243,
        -0.3110564645991377,
        0.6140707236141174,
        0.08158577121790486,
        -1.4806815491355565,
        0.17284131144392148,
        1.0327390973509902,
        0.21572246882942994
      ],
      "row": 0
    },
    {
      "expected_book_back_clearance_m": 0.0813556365585062,
      "expected_book_ceiling_clearance_m": 0.04899326952584504,
      "expected_book_center_world_m": [
        2.893491523394675,
        -0.15308552533390896,
        1.5778599974361471
      ],
      "expected_book_floor_signed_m": 0.000996337534507008,
      "expected_book_shelf_face_clearance_m": -0.21864436344149363,
      "expected_hand_world_position_m": [
        2.8128948235882874,
        -0.154202697623061,
        1.569830811277959
      ],
      "lift_world_z_m": 0.001,
      "outward_world_minus_x_m": 0.0,
      "phase": "micro_lift",
      "q8": [
        0.349999786476243,
        -0.3106537204692976,
        0.617122613363028,
        0.081696222533248,
        -1.476992466051639,
        0.17310387990916742,
        1.032193390885109,
        0.2150984654634221
      ],
      "row": 1
    },
    {
      "expected_book_back_clearance_m": 0.0813556365585062,
      "expected_book_ceiling_clearance_m": 0.04799326952584493,
      "expected_book_center_world_m": [
        2.893491523394675,
        -0.15308552533390896,
        1.5788599974361472
      ],
      "expected_book_floor_signed_m": 0.00199633753450712,
      "expected_book_shelf_face_clearance_m": -0.21864436344149363,
      "expected_hand_world_position_m": [
        2.8128948235882874,
        -0.154202697623061,
        1.5708308112779592
      ],
      "lift_world_z_m": 0.002,
      "outward_world_minus_x_m": 0.0,
      "phase": "micro_lift",
      "q8": [
        0.349999786476243,
        -0.31025110363010006,
        0.6201778005968871,
        0.08180616223726932,
        -1.4732902715907839,
        0.17336587283980986,
        1.0316379550697072,
        0.21447389193293528
      ],
      "row": 2
    },
    {
      "expected_book_back_clearance_m": 0.0813556365585062,
      "expected_book_ceiling_clearance_m": 0.04699326952584504,
      "expected_book_center_world_m": [
        2.893491523394675,
        -0.15308552533390896,
        1.5798599974361471
      ],
      "expected_book_floor_signed_m": 0.00299633753450701,
      "expected_book_shelf_face_clearance_m": -0.21864436344149363,
      "expected_hand_world_position_m": [
        2.8128948235882874,
        -0.154202697623061,
        1.571830811277959
      ],
      "lift_world_z_m": 0.003,
      "outward_world_minus_x_m": 0.0,
      "phase": "micro_lift",
      "q8": [
        0.349999786476243,
        -0.30984861324906093,
        0.6232363076958609,
        0.08191559803053113,
        -1.4695749453081,
        0.17362729888834472,
        1.0310727966361142,
        0.21384874136739376
      ],
      "row": 3
    },
    {
      "expected_book_back_clearance_m": 0.0813556365585062,
      "expected_book_ceiling_clearance_m": 0.045993269525844926,
      "expected_book_center_world_m": [
        2.893491523394675,
        -0.15308552533390896,
        1.5808599974361472
      ],
      "expected_book_floor_signed_m": 0.003996337534507122,
      "expected_book_shelf_face_clearance_m": -0.21864436344149363,
      "expected_hand_world_position_m": [
        2.8128948235882874,
        -0.154202697623061,
        1.5728308112779592
      ],
      "lift_world_z_m": 0.004,
      "outward_world_minus_x_m": 0.0,
      "phase": "micro_lift",
      "q8": [
        0.349999786476243,
        -0.3094462464900623,
        0.6262981913768088,
        0.08202453839211699,
        -1.465846393969101,
        0.1738881745900369,
        1.030497879938731,
        0.21322299266247147
      ],
      "row": 4
    },
    {
      "expected_book_back_clearance_m": 0.0813556365585062,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.893491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.21864436344149363,
      "expected_hand_world_position_m": [
        2.8128948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.0,
      "phase": "micro_lift",
      "q8": [
        0.349999786476243,
        -0.30904400045702474,
        0.6293635091553446,
        0.08213299182424749,
        -1.462104522744736,
        0.17414851655219826,
        1.0299131685637988,
        0.21259662448563346
      ],
      "row": 5
    },
    {
      "expected_book_back_clearance_m": 0.08635563655850609,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8884915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.21364436344149373,
      "expected_hand_world_position_m": [
        2.8078948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.005,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.3069514988064582,
        0.6276862066262057,
        0.08280000400370317,
        -1.4783630151361675,
        0.17523438077842526,
        1.044809908392821,
        0.2126646364833123
      ],
      "row": 6
    },
    {
      "expected_book_back_clearance_m": 0.09135563655850598,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8834915233946754,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.20864436344149384,
      "expected_hand_world_position_m": [
        2.8028948235882876,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.01,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.30483838636397653,
        0.6262450555169756,
        0.08348696074760613,
        -1.4943159294940134,
        0.17635712393094818,
        1.059642411624256,
        0.21271557147036493
      ],
      "row": 7
    },
    {
      "expected_book_back_clearance_m": 0.09635563655850632,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.878491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.2036443634414935,
      "expected_hand_world_position_m": [
        2.7978948235882872,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.015,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.3027046002849538,
        0.6250366541411082,
        0.0841937909741292,
        -1.5099715311789332,
        0.1775159125711334,
        1.0744155070800527,
        0.21275133897231688
      ],
      "row": 8
    },
    {
      "expected_book_back_clearance_m": 0.10135563655850621,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.873491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.1986443634414936,
      "expected_hand_world_position_m": [
        2.7928948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.02,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.30055003536601027,
        0.6240581353261526,
        0.08492043544290288,
        -1.5253371049521383,
        0.17871005600386303,
        1.0891336280258963,
        0.21277366704661632
      ],
      "row": 9
    },
    {
      "expected_book_back_clearance_m": 0.1063556365585061,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8684915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.19364436344149372,
      "expected_hand_world_position_m": [
        2.7878948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.025,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2983745708480485,
        0.6233069043710335,
        0.08566684059610129,
        -1.5404193628480927,
        0.1799389486954071,
        1.1038009097360229,
        0.21278418389087883
      ],
      "row": 10
    },
    {
      "expected_book_back_clearance_m": 0.111355636558506,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8634915233946754,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.18864436344149382,
      "expected_hand_world_position_m": [
        2.7828948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.03,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.29617807220456077,
        0.622780614017035,
        0.08643295868379156,
        -1.5552244901469712,
        0.18120206174671888,
        1.1184212102292148,
        0.2127844275978438
      ],
      "row": 11
    },
    {
      "expected_book_back_clearance_m": 0.11635563655850634,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.858491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.18364436344149349,
      "expected_hand_world_position_m": [
        2.777894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.035,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.29396039284984016,
        0.6224771417910271,
        0.08721874790451616,
        -1.5697581865159285,
        0.18249893532199718,
        1.1329981285705186,
        0.21277585475928654
      ],
      "row": 12
    },
    {
      "expected_book_back_clearance_m": 0.12135563655850623,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.853491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.1786443634414936,
      "expected_hand_world_position_m": [
        2.7728948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.04,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2917213757469029,
        0.6223945694085296,
        0.08802417252181512,
        -1.5840257029313978,
        0.18382917190714634,
        1.147535021034841,
        0.2127598480821055
      ],
      "row": 13
    },
    {
      "expected_book_back_clearance_m": 0.12635563655850612,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8484915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.1736443634414937,
      "expected_hand_world_position_m": [
        2.7678948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.045,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2894608549597058,
        0.6225311639695862,
        0.08884920300096669,
        -1.5980318749127975,
        0.1851924302672064,
        1.1620350153943342,
        0.21273772312649034
      ],
      "row": 14
    },
    {
      "expected_book_back_clearance_m": 0.13135563655850602,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8434915233946754,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.1686443634414938,
      "expected_hand_world_position_m": [
        2.7628948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.05,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2871786571395194,
        0.6228853607178904,
        0.08969381614151854,
        -1.6117811525196064,
        0.18658842000964618,
        1.176501023549322,
        0.21271073428261525
      ],
      "row": 15
    },
    {
      "expected_book_back_clearance_m": 0.13635563655850635,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.838491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.16364436344149347,
      "expected_hand_world_position_m": [
        2.757894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.055,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2848746029616022,
        0.6234557471651105,
        0.09055799521539858,
        -1.6252776275041585,
        0.18801689666419424,
        1.1909357526954834,
        0.21268008007346315
      ],
      "row": 16
    },
    {
      "expected_book_back_clearance_m": 0.14135563655850625,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.833491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.15864436344149357,
      "expected_hand_world_position_m": [
        2.7528948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.06,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.28254850851369523,
        0.6242410484109896,
        0.09144173010399394,
        -1.6385250579567143,
        0.18947765720778073,
        1.2053417151929267,
        0.2126469078630841
      ],
      "row": 17
    },
    {
      "expected_book_back_clearance_m": 0.14635563655850614,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8284915233946752,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.15364436344149368,
      "expected_hand_world_position_m": [
        2.7478948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.065,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.28020018665592716,
        0.6252401135126268,
        0.09234501745000272,
        -1.6515268907361855,
        0.1909705359653305,
        1.219721237283078,
        0.21261231803011854
      ],
      "row": 18
    },
    {
      "expected_book_back_clearance_m": 0.15135563655850603,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8234915233946754,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.1486443634414938,
      "expected_hand_world_position_m": [
        2.7428948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.07,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2778294483386326,
        0.6264519027751584,
        0.0932678608002196,
        -1.6642862819410433,
        0.1924954008384175,
        1.2340764667786546,
        0.21257736767181554
      ],
      "row": 19
    },
    {
      "expected_book_back_clearance_m": 0.15635563655850637,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.818491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.14364436344149345,
      "expected_hand_world_position_m": [
        2.737894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.075,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2754361039043141,
        0.6278754758547416,
        0.0942102707650526,
        -1.676806115642624,
        0.19405214980510635,
        1.2484093798396874,
        0.21254307387615
      ],
      "row": 20
    },
    {
      "expected_book_back_clearance_m": 0.16135563655850627,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.813491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.13864436344149356,
      "expected_hand_world_position_m": [
        2.7328948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.08,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.27301996436556275,
        0.6295099805777369,
        0.09517226517850819,
        -1.6890890210758605,
        0.19564070765315153,
        1.2627217869335607,
        0.21251041660933612
      ],
      "row": 21
    },
    {
      "expected_book_back_clearance_m": 0.16635563655850616,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8084915233946752,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.13364436344149366,
      "expected_hand_world_position_m": [
        2.7278948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.085,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2705808426618666,
        0.6313546423930392,
        0.09615386925707615,
        -1.7011373884578294,
        0.19726102290953917,
        1.277015338065905,
        0.21248034125410264
      ],
      "row": 22
    },
    {
      "expected_book_back_clearance_m": 0.17135563655850605,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.8034915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.12864436344149377,
      "expected_hand_world_position_m": [
        2.7228948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.09,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2681185549125421,
        0.633408754386425,
        0.09715511577337438,
        -1.7129533835850164,
        0.19891306492734415,
        1.2912915273622017,
        0.2124537608222047
      ],
      "row": 23
    },
    {
      "expected_book_back_clearance_m": 0.1763556365585064,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.798491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.12364436344149343,
      "expected_hand_world_position_m": [
        2.717894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.095,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2656329216517152,
        0.6356716677935955,
        0.09817604522294042,
        -1.7245389613417776,
        0.2005968211069136,
        1.3055516970685561,
        0.2124315578751478
      ],
      "row": 24
    },
    {
      "expected_book_back_clearance_m": 0.18135563655850628,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.793491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.11864436344149354,
      "expected_hand_world_position_m": [
        2.7128948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.1,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.26312376905215146,
        0.6381427829584666,
        0.09921670598859299,
        -1.7358958782372775,
        0.20231229422290278,
        1.3197970410353663,
        0.21241458617287012
      ],
      "row": 25
    },
    {
      "expected_book_back_clearance_m": 0.18635563655850618,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.788491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.11364436344149365,
      "expected_hand_world_position_m": [
        2.7078948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.105,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2605909301486732,
        0.6408215406899851,
        0.10027715451134714,
        -1.7470257040760508,
        0.20405949982958022,
        1.3340286077426247,
        0.2124036720652338
      ],
      "row": 26
    },
    {
      "expected_book_back_clearance_m": 0.19135563655850607,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.7834915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.10864436344149375,
      "expected_hand_world_position_m": [
        2.7028948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.11,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.25803424604708364,
        0.6437074139777477,
        0.10135745544791516,
        -1.7579298328533586,
        0.20583846372885387,
        1.3482473029177493,
        0.21239961564970075
      ],
      "row": 27
    },
    {
      "expected_book_back_clearance_m": 0.1963556365585064,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.778491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.10364436344149341,
      "expected_hand_world_position_m": [
        2.697894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.115,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.25545356713052364,
        0.6467999000315499,
        0.10245768182574777,
        -1.7686094929595901,
        0.20764921947684659,
        1.3624538917957312,
        0.2124031917041519
      ],
      "row": 28
    },
    {
      "expected_book_back_clearance_m": 0.2013556365585063,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.773491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.09864436344149352,
      "expected_hand_world_position_m": [
        2.6928948235882872,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.12,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2528487542542325,
        0.6500985126169517,
        0.10357791518257449,
        -1.7790657567656316,
        0.20949180591488373,
        1.3766490010653751,
        0.2124151504109121
      ],
      "row": 29
    },
    {
      "expected_book_back_clearance_m": 0.2063556365585062,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.768491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.09364436344149363,
      "expected_hand_world_position_m": [
        2.6878948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.125,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.25021967993318456,
        0.6536027746618348,
        0.10471824569315402,
        -1.7892995496562834,
        0.21136626470672362,
        1.390833120543941,
        0.21243621788031686
      ],
      "row": 30
    },
    {
      "expected_book_back_clearance_m": 0.2113556365585061,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.7634915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.08864436344149373,
      "expected_hand_world_position_m": [
        2.6828948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.13,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2475662295153806,
        0.6573122111147411,
        0.10587877227290544,
        -1.7993116585695088,
        0.2132726378698706,
        1.4050066046186844,
        0.21246709648574175
      ],
      "row": 31
    },
    {
      "expected_book_back_clearance_m": 0.21635563655850598,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.7584915233946754,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.08364436344149384,
      "expected_hand_world_position_m": [
        2.6778948235882876,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.135,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.24488830234176337,
        0.6612263420380204,
        0.10705960265782453,
        -1.8091027400954283,
        0.2152109652866626,
        1.419169673492338,
        0.21250846501685702
      ],
      "row": 32
    },
    {
      "expected_book_back_clearance_m": 0.22135563655850632,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.753491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.0786443634414935,
      "expected_hand_world_position_m": [
        2.6728948235882872,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.14,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.24218581288303326,
        0.6653446759241831,
        0.10826085344812499,
        -1.818673328180766,
        0.21718128218640492,
        1.4333224142665337,
        0.21256097866177182
      ],
      "row": 33
    },
    {
      "expected_book_back_clearance_m": 0.2263556365585062,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.748491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.07364436344149361,
      "expected_hand_world_position_m": [
        2.6678948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.145,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2394586918582244,
        0.6696667032257536,
        0.10948265012031307,
        -1.8280238414821903,
        0.21918361658465216,
        1.4474647818972062,
        0.21262526882071067
      ],
      "row": 34
    },
    {
      "expected_book_back_clearance_m": 0.2313556365585061,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.7434915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.06864436344149372,
      "expected_hand_world_position_m": [
        2.6628948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.15,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.23670688732049097,
        0.6741918900920131,
        0.11072512699032118,
        -1.8371545904060835,
        0.22121798667515,
        1.4615966000524678,
        0.2127019427625173
      ],
      "row": 35
    },
    {
      "expected_book_back_clearance_m": 0.236355636558506,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.7384915233946754,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.06364436344149382,
      "expected_hand_world_position_m": [
        2.6578948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.155,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.23393036570559825,
        0.6789196723089661,
        0.11198842712249918,
        -1.8460657838681906,
        0.22328439816687443,
        1.4757175619026814,
        0.2127915831297406
      ],
      "row": 36
    },
    {
      "expected_book_back_clearance_m": 0.24135563655850634,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.733491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.058644363441493486,
      "expected_hand_world_position_m": [
        2.652894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.16,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.23112911284195045,
        0.6838494494416102,
        0.1132727021834508,
        -1.8547575358041177,
        0.22538284155794175,
        1.4898272308728273,
        0.21289474729612698
      ],
      "row": 37
    },
    {
      "expected_book_back_clearance_m": 0.24635563655850623,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.728491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.05364436344149359,
      "expected_hand_world_position_m": [
        2.6478948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.165,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2283031349063623,
        0.6889805791789364,
        0.11457811222366202,
        -1.863229871456887,
        0.22751328934622353,
        1.5039250413830696,
        0.2130119665868764
      ],
      "row": 38
    },
    {
      "expected_book_back_clearance_m": 0.2513556365585061,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.7234915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.0486443634414937,
      "expected_hand_world_position_m": [
        2.6428948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.17,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.22545245932172178,
        0.6943123718848507,
        0.11590482538391329,
        -1.8714827334659854,
        0.22967569317222222,
        1.5180102996050517,
        0.2131437453665659
      ],
      "row": 39
    },
    {
      "expected_book_back_clearance_m": 0.256355636558506,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.7184915233946754,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.043644363441493805,
      "expected_hand_world_position_m": [
        2.6378948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.17500000000000002,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.22257713558628375,
        0.6998440853597823,
        0.11725301751702581,
        -1.8795159877786545,
        0.23186998089413877,
        1.5320821842586814,
        0.21329056000227242
      ],
      "row": 40
    },
    {
      "expected_book_back_clearance_m": 0.26135563655850635,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.713491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.03864436344149347,
      "expected_hand_world_position_m": [
        2.632894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.18,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.21967723602030734,
        0.7055749198188637,
        0.11862287171155919,
        -1.887329429402163,
        0.23409605359869387,
        1.546139747473278,
        0.21345285771213515
      ],
      "row": 41
    },
    {
      "expected_book_back_clearance_m": 0.26635563655850625,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.708491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.033644363441493574,
      "expected_hand_world_position_m": [
        2.6278948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.185,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.21675285642353936,
        0.7115040130950933,
        0.1200145777136589,
        -1.8949227880127868,
        0.2363537825490485,
        1.5601819157364354,
        0.2136310553057903
      ],
      "row": 42
    },
    {
      "expected_book_back_clearance_m": 0.27135563655850614,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.7034915233946752,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.02864436344149368,
      "expected_hand_world_position_m": [
        2.6228948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.19,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.21380411663471605,
        0.7176304360761281,
        0.12142833124141064,
        -1.9022957334358217,
        0.23864300607416597,
        1.5742074909526658,
        0.21382553782471184
      ],
      "row": 43
    },
    {
      "expected_book_back_clearance_m": 0.27635563655850603,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.6984915233946754,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.023644363441493788,
      "expected_hand_world_position_m": [
        2.6178948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.195,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.21083116097082097,
        0.7239531883844819,
        0.12286433317205486,
        -1.909447881007754,
        0.24096352641259322,
        1.588215151631121,
        0.21403665709816136
      ],
      "row": 44
    },
    {
      "expected_book_back_clearance_m": 0.2813556365585064,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.693491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.01864436344149345,
      "expected_hand_world_position_m": [
        2.612894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.2,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2078341585435161,
        0.7304711943121521,
        0.12432278860529192,
        -1.9163787968308277,
        0.2433151065162636,
        1.6022034542225192,
        0.214264730221181
      ],
      "row": 45
    },
    {
      "expected_book_back_clearance_m": 0.28635563655850627,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.688491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.013644363441493557,
      "expected_hand_world_position_m": [
        2.6078948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.20500000000000002,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.20481330343492385,
        0.7371832990211572,
        0.1258039057898897,
        -1.9230880029270576,
        0.24569746682941354,
        1.61617083462203,
        0.21451003796954862
      ],
      "row": 46
    },
    {
      "expected_book_back_clearance_m": 0.29135563655850616,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.6834915233946752,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.008644363441493663,
      "expected_hand_world_position_m": [
        2.6028948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.21,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.2017688147236045,
        0.7440882650216757,
        0.12730789491162817,
        -1.9295749822984878,
        0.24811028205643676,
        1.6301156098549974,
        0.21477282316360352
      ],
      "row": 47
    },
    {
      "expected_book_back_clearance_m": 0.29635563655850605,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.6784915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": -0.00364436344149377,
      "expected_hand_world_position_m": [
        2.5978948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.215,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.1987009363425727,
        0.7511847689397527,
        0.12883496673174058,
        -1.9358391838972542,
        0.25055317793926374,
        1.6440359799586022,
        0.21505328899836051
      ],
      "row": 48
    },
    {
      "expected_book_back_clearance_m": 0.3013556365585064,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.673491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": 0.0013556365585065677,
      "expected_hand_world_position_m": [
        2.592894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.22,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.19560993676337085,
        0.7584713985865738,
        0.1303853310794891,
        -1.941880027508607,
        0.2530257280616296,
        1.6579300300727793,
        0.21535159735244144
      ],
      "row": 49
    },
    {
      "expected_book_back_clearance_m": 0.3063556365585063,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.668491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": 0.006355636558506461,
      "expected_hand_world_position_m": [
        2.5878948235882873,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.225,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.19249610849098897,
        0.7659466503418683,
        0.13195919519364996,
        -1.9476969085471607,
        0.2555274507051422,
        1.6717957327507142,
        0.2156678670944718
      ],
      "row": 50
    },
    {
      "expected_book_back_clearance_m": 0.3113556365585062,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.663491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": 0.011355636558506355,
      "expected_hand_world_position_m": [
        2.5828948235882874,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.23,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.1893597673609814,
        0.7736089268614079,
        0.13355676191580823,
        -1.9532892027675672,
        0.2580578057815139,
        1.6856309504974765,
        0.21600217240362757
      ],
      "row": 51
    },
    {
      "expected_book_back_clearance_m": 0.31635563655850607,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.6584915233946753,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": 0.016355636558506248,
      "expected_hand_world_position_m": [
        2.5778948235882875,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.23500000000000001,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.1862012516244889,
        0.7814565351209644,
        0.13517822773369306,
        -1.9586562708855793,
        0.2606161918714897,
        1.6994334385424186,
        0.21635454112540653
      ],
      "row": 52
    },
    {
      "expected_book_back_clearance_m": 0.3213556365585064,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.653491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": 0.021355636558506585,
      "expected_hand_world_position_m": [
        2.572894823588287,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.24,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.18302092082023746,
        0.7894876848059305,
        0.13682378068795248,
        -1.9637974631083674,
        0.2632019433963345,
        1.7132008478507934,
        0.21672495317804416
      ],
      "row": 53
    },
    {
      "expected_book_back_clearance_m": 0.3263556365585063,
      "expected_book_ceiling_clearance_m": 0.04499326952584504,
      "expected_book_center_world_m": [
        2.648491523394675,
        -0.15308552533390896,
        1.5818599974361471
      ],
      "expected_book_floor_signed_m": 0.004996337534507012,
      "expected_book_shelf_face_clearance_m": 0.02635563655850648,
      "expected_hand_world_position_m": [
        2.5678948235882872,
        -0.154202697623061,
        1.573830811277959
      ],
      "lift_world_z_m": 0.005,
      "outward_world_minus_x_m": 0.245,
      "phase": "outward",
      "q8": [
        0.349999786476243,
        -0.1798191544203312,
        0.7977004870554982,
        0.13849359814364273,
        -1.9687121235690639,
        0.2658143279576978,
        1.7269307283755084,
        0.21711333903326963
      ],
      "row": 54
    }
  ],
  "route_q8_sha256": "6d8caebaa62da321d5f2db3e7e04d0a45d2f7e845ffab9afa0abb3a1b745f54c"
}


'''
CERTIFICATE: Mapping[str, object] = json.loads(CERTIFICATE_JSON)

CERTIFICATE_KIND = str(CERTIFICATE['kind'])
ROUTE_Q8_SHA256 = str(CERTIFICATE['route_q8_sha256'])

CHECKPOINT = CERTIFICATE['input']
AUDIT = CERTIFICATE['audit']
EXECUTION_POLICY = CERTIFICATE['execution_policy']

CHECKPOINT_BASE_WORLD_XYYAW = tuple(CHECKPOINT['base_world_xyyaw'])
CHECKPOINT_LEFT_Q8 = tuple(CHECKPOINT['left_q8_measured'])
CHECKPOINT_RIGHT_Q7 = tuple(CHECKPOINT['right_q7_measured'])
CHECKPOINT_HEAD_Q2 = tuple(CHECKPOINT['head_q2_measured'])
CHECKPOINT_BOOK_POSITION_WORLD_M = tuple(CHECKPOINT['book_position_world_m'])
CHECKPOINT_BOOK_QUATERNION_XYZW = tuple(CHECKPOINT['book_quaternion_xyzw'])
CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M = tuple(
    tuple(row) for row in CHECKPOINT['book_physical_bounds_world_m']
)
CHECKPOINT_GRIPPER_MASTER_M = float(CHECKPOINT['gripper_master_m'])

ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP = tuple(
    tuple(row) for row in CERTIFICATE['attached_book_corners_physical_in_grasp']
)
ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP = tuple(
    tuple(row)
    for row in CERTIFICATE['attached_book_corners_padding_15mm_in_grasp']
)

RECOVERY_ROUTE = tuple(CERTIFICATE['route'])
LOADED_RECOVERY_Q8 = tuple(
    tuple(row['q8']) for row in RECOVERY_ROUTE
)
RECOVERY_START_Q8 = LOADED_RECOVERY_Q8[0]

MICRO_LIFT_COUNT = 5
MICRO_LIFT_STEP_M = 0.001
MICRO_LIFT_TOTAL_M = MICRO_LIFT_COUNT * MICRO_LIFT_STEP_M
MICRO_LIFT_Q8 = LOADED_RECOVERY_Q8[1:1 + MICRO_LIFT_COUNT]

OUTWARD_COUNT = 49
OUTWARD_STEP_M = 0.005
OUTWARD_TOTAL_M = OUTWARD_COUNT * OUTWARD_STEP_M
OUTWARD_Q8 = LOADED_RECOVERY_Q8[1 + MICRO_LIFT_COUNT:]

MINIMUM_MICRO_LIFT_DURATION_S = float(
    EXECUTION_POLICY['minimum_micro_lift_duration_s']
)
MINIMUM_OUTWARD_DURATION_S = float(
    EXECUTION_POLICY['minimum_outward_duration_s']
)
MAXIMUM_RELATIVE_TRANSLATION_SLIP_PER_LEG_M = float(
    EXECUTION_POLICY['maximum_relative_translation_slip_per_leg_m']
)
MAXIMUM_RELATIVE_ROTATION_SLIP_PER_LEG_RAD = float(
    EXECUTION_POLICY['maximum_relative_rotation_slip_per_leg_rad']
)


def _canonical_certificate_bytes() -> bytes:
    return json.dumps(
        CERTIFICATE,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')


def semantic_digest() -> str:
    """Return the deterministic digest of the embedded certificate data."""

    return hashlib.sha256(_canonical_certificate_bytes()).hexdigest()


def source_json_digest() -> str:
    """Return the digest of the exact embedded source JSON text."""

    return hashlib.sha256(CERTIFICATE_JSON.encode('utf-8')).hexdigest()


def route_q8_digest() -> str:
    """Return the deterministic digest of the exact 55-row q8 route."""

    payload = json.dumps(
        [list(row) for row in LOADED_RECOVERY_Q8],
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _finite_vector(values: object, length: int) -> bool:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return False
    return len(vector) == length and all(math.isfinite(value) for value in vector)


def validate_certificate() -> Tuple[bool, Mapping[str, float]]:
    """Fail closed if any embedded checkpoint, phase, or audit fact changed."""

    route_rows_valid = all(
        isinstance(row, Mapping)
        and int(row.get('row', -1)) == index
        and _finite_vector(row.get('q8'), 8)
        for index, row in enumerate(RECOVERY_ROUTE)
    )
    phase_rows_valid = bool(
        len(RECOVERY_ROUTE) == 1 + MICRO_LIFT_COUNT + OUTWARD_COUNT
        and RECOVERY_ROUTE[0]['phase'] == 'start'
        and all(
            RECOVERY_ROUTE[index]['phase'] == 'micro_lift'
            and math.isclose(
                float(RECOVERY_ROUTE[index]['lift_world_z_m']),
                MICRO_LIFT_STEP_M * index,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(RECOVERY_ROUTE[index]['outward_world_minus_x_m']),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for index in range(1, 1 + MICRO_LIFT_COUNT)
        )
        and all(
            RECOVERY_ROUTE[index]['phase'] == 'outward'
            and math.isclose(
                float(RECOVERY_ROUTE[index]['lift_world_z_m']),
                MICRO_LIFT_TOTAL_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(RECOVERY_ROUTE[index]['outward_world_minus_x_m']),
                OUTWARD_STEP_M * (index - MICRO_LIFT_COUNT),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for index in range(1 + MICRO_LIFT_COUNT, len(RECOVERY_ROUTE))
        )
    )
    checkpoint_valid = bool(
        _finite_vector(CHECKPOINT_BASE_WORLD_XYYAW, 3)
        and _finite_vector(CHECKPOINT_LEFT_Q8, 8)
        and _finite_vector(CHECKPOINT_RIGHT_Q7, 7)
        and _finite_vector(CHECKPOINT_HEAD_Q2, 2)
        and _finite_vector(CHECKPOINT_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(CHECKPOINT_BOOK_QUATERNION_XYZW, 4)
        and math.isfinite(CHECKPOINT_GRIPPER_MASTER_M)
        and CHECKPOINT_LEFT_Q8 == RECOVERY_START_Q8
    )
    corners_valid = bool(
        len(ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP) == 8
        and len(ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP) == 8
        and all(
            _finite_vector(row, 3)
            for row in (
                *ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP,
                *ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP,
            )
        )
    )
    torso_is_fixed = all(
        math.isclose(
            row[0], RECOVERY_START_Q8[0], rel_tol=0.0, abs_tol=1e-12
        )
        for row in LOADED_RECOVERY_Q8
    )
    maximum_endpoint_delta = max(
        max(abs(after - before) for before, after in zip(first[1:], second[1:]))
        for first, second in zip(
            LOADED_RECOVERY_Q8, LOADED_RECOVERY_Q8[1:]
        )
    )
    audit_valid = bool(
        AUDIT.get('collision_free') is True
        and int(AUDIT.get('dense_samples', 0)) == 394
        and float(AUDIT.get('maximum_joint_increment_rad', math.nan)) == 0.002
        and min(
            float(AUDIT.get('minimum_self_aabb_clearance_m', math.nan)),
            float(
                AUDIT.get(
                    'minimum_15mm_padded_payload_robot_aabb_clearance_m',
                    math.nan,
                )
            ),
            float(
                AUDIT.get(
                    'minimum_closed_gripper_shelf_triangle_aabb_clearance_m',
                    math.nan,
                )
            ),
            float(
                AUDIT.get('minimum_book_ceiling_clearance_m', math.nan)
            ),
            float(AUDIT.get('minimum_book_back_clearance_m', math.nan)),
            float(
                AUDIT.get('final_book_shelf_face_clearance_m', math.nan)
            ),
        ) > 0.0
        and AUDIT.get('expected_support_floor_contact_only_until_lift') is True
    )
    execution_policy_valid = bool(
        EXECUTION_POLICY.get('keep_current_gripper_pressure') is True
        and EXECUTION_POLICY.get('require_fresh_bilateral_contact_each_leg')
        is True
        and int(EXECUTION_POLICY.get('micro_lift_count', -1))
        == MICRO_LIFT_COUNT
        and int(EXECUTION_POLICY.get('outward_count', -1)) == OUTWARD_COUNT
        and math.isclose(
            float(EXECUTION_POLICY.get('micro_lift_step_m', math.nan)),
            MICRO_LIFT_STEP_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(EXECUTION_POLICY.get('outward_step_m', math.nan)),
            OUTWARD_STEP_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and MINIMUM_MICRO_LIFT_DURATION_S >= 0.8
        and MINIMUM_OUTWARD_DURATION_S >= 1.0
        and MAXIMUM_RELATIVE_TRANSLATION_SLIP_PER_LEG_M <= 0.0005
        and MAXIMUM_RELATIVE_ROTATION_SLIP_PER_LEG_RAD <= 0.002
    )
    valid = bool(
        CERTIFICATE_KIND
        == 'read_only_partial_step2_loaded_recovery_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and route_rows_valid
        and phase_rows_valid
        and checkpoint_valid
        and corners_valid
        and torso_is_fixed
        and audit_valid
        and execution_policy_valid
    )
    return valid, {
        'route_rows': float(len(LOADED_RECOVERY_Q8)),
        'micro_lift_rows': float(len(MICRO_LIFT_Q8)),
        'outward_rows': float(len(OUTWARD_Q8)),
        'maximum_endpoint_delta_rad': maximum_endpoint_delta,
        'final_shelf_clearance_m': float(
            AUDIT['final_book_shelf_face_clearance_m']
        ),
        'minimum_audit_margin_m': min(
            float(AUDIT['minimum_self_aabb_clearance_m']),
            float(
                AUDIT[
                    'minimum_15mm_padded_payload_robot_aabb_clearance_m'
                ]
            ),
            float(
                AUDIT[
                    'minimum_closed_gripper_shelf_triangle_aabb_clearance_m'
                ]
            ),
        ),
    }


__all__ = (
    'ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP',
    'ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP',
    'AUDIT',
    'CERTIFICATE',
    'CERTIFICATE_KIND',
    'CERTIFICATE_ORIGINAL_JSON_FILE_SHA256',
    'CERTIFICATE_SEMANTIC_DIGEST',
    'CERTIFICATE_SOURCE_JSON_SHA256',
    'CHECKPOINT_BASE_WORLD_XYYAW',
    'CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M',
    'CHECKPOINT_BOOK_POSITION_WORLD_M',
    'CHECKPOINT_BOOK_QUATERNION_XYZW',
    'CHECKPOINT_GRIPPER_MASTER_M',
    'CHECKPOINT_HEAD_Q2',
    'CHECKPOINT_LEFT_Q8',
    'CHECKPOINT_RIGHT_Q7',
    'EXECUTION_POLICY',
    'LOADED_RECOVERY_Q8',
    'MAXIMUM_RELATIVE_ROTATION_SLIP_PER_LEG_RAD',
    'MAXIMUM_RELATIVE_TRANSLATION_SLIP_PER_LEG_M',
    'MICRO_LIFT_COUNT',
    'MICRO_LIFT_Q8',
    'MICRO_LIFT_STEP_M',
    'MICRO_LIFT_TOTAL_M',
    'MINIMUM_MICRO_LIFT_DURATION_S',
    'MINIMUM_OUTWARD_DURATION_S',
    'OUTWARD_COUNT',
    'OUTWARD_Q8',
    'OUTWARD_STEP_M',
    'OUTWARD_TOTAL_M',
    'RECOVERY_ROUTE',
    'RECOVERY_START_Q8',
    'ROUTE_Q8_SHA256',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
