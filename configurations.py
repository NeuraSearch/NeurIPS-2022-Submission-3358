# coding:utf-8

from argparse import ArgumentParser

CONST_LIST = ["const_1", "const_2", "const_3", "const_4", "const_5", "const_6", "const_7", "const_8", "const_9", "const_10",
              "const_100", "const_1000", "const_10000", "const_100000", "const_1000000", "const_10000000",
              "const_1000000000", "const_m1", "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", 
              "none"]

OPERATION_LIST = ["add", "subtract", "multiply", "divide", "exp", "greater", 
                  "table_sum", "table_average", "table_max", "table_min",
                  "EOF"]

OPERATION_LIST_DROP = ["add", "subtract", "multiply", "divide", "greater", "biggest", "smallest", "EOF"]

CONST_LIST_MATHQA = ["const_pi", "const_2", "const_1", "const_3", "const_4", "const_6", "const_10", "const_12", "const_100", "const_1000",
                     "const_60", "const_3600", "const_1.6", "const_0.6", "const_0.2778", "const_0.3937", "const_2.54",
                     "const_0.4535", "const_2.2046", "const_3.6", "const_deg_to_rad", "const_180", "const_0.25", "const_0.33",
                     "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", "#11", "#12", "#13", "#14", "#15", "#16",
                     "none"]

CONST_LIST_MATHQA_2 = ["const_pi", "const_2", "const_1", "const_3", "const_4", "const_6", "const_10", "const_12", "const_100", "const_1000",
                     "const_60", "const_3600", "const_1.6", "const_0.6", "const_0.2778", "const_0.3937", "const_2.54",
                     "const_0.4535", "const_2.2046", "const_3.6", "const_deg_to_rad", "const_180", "const_0.25", "const_0.33",
                     "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", "#11", "#12", "#13", "#14", "#15", "#16",
                     "none"]

OPERATION_LIST_MATHQA = ["add", "subtract", "multiply", "divide", "gcd", "lcm", "power", "max", "min",
                         "reminder", "round", "radians_to_degress", "degree_to_radians",
                         "sum_consecutive_number", "circle_arc", "semi_circle_perimiter", "circle_sector_area",
                         "rectangle_perimeter", "rectangle_area", "trapezium_area",
                         "rhombus_area", "quadrilateral_area", "volume_cone", "volume_rectangular_prism",
                         "volume_cylinder", "surface_cone", "surface_cylinder", "surface_rectangular_prism",
                         "side_by_diagonal", "diagonal", "triangle_perimeter",
                         "triangle_area", "triangle_area_three_edges", "union_prob", "combination", "permutation", "count_interval",
                         "percent", "p_after_gain", "p_after_loss", "price_after_gain", "price_after_loss", "from_percent", "gain_percent",
                         "loss_percent", "negate_percent", "original_price_before_gain", "original_price_before_loss", "to_percent", "speed",
                         "combined_work", "find_work", "speed_ratio_steel_to_stream", "speed_in_still_water", "stream_speed", 
                         "floor", "cosine",
                         "cube_edge_by_volume", "volume_cube", "sine", "factorial", "square_area", "negate", "sqrt", "circle_area",  "surface_sphere",
                         "log", "surface_cube", "rhombus_perimeter", "volume_sphere", "tangent", "square_perimeter", "circumface", "square_edge_by_area",
                         "inverse", "square_edge_by_perimeter", "negate_prob",
                         "EOF"]

OPERATION_LIST_MATHQA_2 = ["add", "subtract", "multiply", "divide", "gcd", "lcm", "power", "max", "min",
                         "reminder", "round", "radians_to_degress", "degree_to_radians",
                         "sum_consecutive_number", "circle_arc", "semi_circle_perimiter", "circle_sector_area",
                         "rectangle_perimeter", "rectangle_area", "trapezium_area",
                         "rhombus_area", "quadrilateral_area", "volume_cone", "volume_rectangular_prism",
                         "volume_cylinder", "surface_cone", "surface_cylinder", "surface_rectangular_prism",
                         "side_by_diagonal", "diagonal", "triangle_perimeter",
                         "triangle_area", "triangle_area_three_edges", "union_prob", "combination", "permutation", "count_interval",
                         "percent", "p_after_gain", "p_after_loss", "price_after_gain", "price_after_loss", "from_percent", "gain_percent",
                         "loss_percent", "negate_percent", "original_price_before_gain", "original_price_before_loss", "to_percent", "speed",
                         "combined_work", "find_work", "speed_ratio_steel_to_stream", "speed_in_still_water", "stream_speed", 
                         "floor", "cosine",
                         "cube_edge_by_volume", "volume_cube", "sine", "factorial", "square_area", "negate", "sqrt", "circle_area",  "surface_sphere",
                         "log", "surface_cube", "rhombus_perimeter", "volume_sphere", "tangent", "square_perimeter", "circumface", "square_edge_by_area",
                         "inverse", "square_edge_by_perimeter", "negate_prob",
                         "GO", ")", "EOF"]

CONST_LIST_JOINED = ["const_pi", "const_1", "const_2", "const_3", "const_4", "const_5", "const_6", "const_7", "const_8", "const_9", "const_10", "const_12", "const_100", "const_1000",
                     "const_1000", "const_10000", "const_100000", "const_1000000", "const_10000000", "const_1000000000", "const_m1", 
                     "const_60", "const_3600", "const_1.6", "const_0.6", "const_0.2778", "const_0.3937", "const_2.54",
                     "const_0.4535", "const_2.2046", "const_3.6", "const_deg_to_rad", "const_180", "const_0.25", "const_0.33",
                     "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", "#11", "#12", "#13", "#14", "#15", "#16",
                     "none"]

OPERATION_LIST_JOINED = ["add", "subtract", "multiply", "divide", "gcd", "lcm", "power", "max", "min", "exp", "greater", 
                         "table_sum", "table_average", "table_max", "table_min",
                         "reminder", "round", "radians_to_degress", "degree_to_radians",
                         "sum_consecutive_number", "circle_arc", "semi_circle_perimiter", "circle_sector_area",
                         "rectangle_perimeter", "rectangle_area", "trapezium_area",
                         "rhombus_area", "quadrilateral_area", "volume_cone", "volume_rectangular_prism",
                         "volume_cylinder", "surface_cone", "surface_cylinder", "surface_rectangular_prism",
                         "side_by_diagonal", "diagonal", "triangle_perimeter",
                         "triangle_area", "triangle_area_three_edges", "union_prob", "combination", "permutation", "count_interval",
                         "percent", "p_after_gain", "p_after_loss", "price_after_gain", "price_after_loss", "from_percent", "gain_percent",
                         "loss_percent", "negate_percent", "original_price_before_gain", "original_price_before_loss", "to_percent", "speed",
                         "combined_work", "find_work", "speed_ratio_steel_to_stream", "speed_in_still_water", "stream_speed", 
                         "floor", "cosine",
                         "cube_edge_by_volume", "volume_cube", "sine", "factorial", "square_area", "negate", "sqrt", "circle_area",  "surface_sphere",
                         "log", "surface_cube", "rhombus_perimeter", "volume_sphere", "tangent", "square_perimeter", "circumface", "square_edge_by_area",
                         "inverse", "square_edge_by_perimeter", "negate_prob",
                         "EOF"]

OPERATION_LIST_DROP_ANNOTATED = ['biggest', 'subtract', 'add', 'multiply', 'divide', 'smallest', 'secondsmallest', 'EOF']
CONST_LIST_DROP_ANNOTATED = ['const_0', 'const_1', 'const_100', 'const_1000000',  'const_1000000000', '#0', '#1', '#2', '#3', '#4', "#5", "#6", "none"]

def add_path_relevant_args(parser: ArgumentParser):
    parser.add_argument("--data_name", type=str, choices=["finqa", "mathqa", "drop", "svamp", "drop_annotated", "drop_fewshot", "drop_fakedata"],
                        help="the selected data name.")

    parser.add_argument("--train_data", type=str,
                        help="the path for the train data with retrieved results.")
    parser.add_argument("--dev_data", type=str,
                        help="the path for the dev data with retrieved results.")
    parser.add_argument("--test_data", type=str,
                        help="the path for the test data with retrieved results.")
    parser.add_argument("--save_dir", type=str,
                        help="the directory to saved the processed data.")
    
    parser.add_argument("--cached_train_data", type=str,
                        help="the path for the processed train data cache.")
    parser.add_argument("--cached_dev_data", type=str,
                        help="the path for the processed dev data cache.")
    parser.add_argument("--cached_test_data", type=str,
                        help="the path for the processed test data cache.")

    parser.add_argument("--model_save_dir", type=str,
                        help="the directory where model saved.")
    parser.add_argument("--eval_results_dir", type=str,
                        help="the directory for saving the evaluation results.")
    
    parser.add_argument("--reload_model_path", type=str,
                        help="the reload model path.")
    parser.add_argument("--reload_config_path", type=str,
                        help="the reload configuraion path.")
    parser.add_argument("--reload_optimizer_path", type=str,
                        help="the reload optimizer path.")
    parser.add_argument("--reload_scheduler_path", type=str,
                        help="the reload learning rate scheduler path.")

def add_model_relevant_args(parser: ArgumentParser):
    parser.add_argument("--plm", type=str, choices=["roberta-large", "roberta-base", "t5-small", "t5-large"],
                        help="the selection for the pretrained model in transformers.")

    parser.add_argument("--n_layers", type=int,
                        help="the number of layers for the RNN.")

    parser.add_argument("--dropout_p", type=float,
                        help="the probability for dropout.")

    parser.add_argument("--max_op_len", type=int,
                        help="maximize number for the operator")

    parser.add_argument("--max_argu_len", type=int,
                        help="maximize number for the arguments for each operator")

    parser.add_argument("--merge_op", type=int,
                        help="whether merge the same consecutive operators.")

    parser.add_argument("--n_head", type=int,
                        help="number of GNN head.")

    parser.add_argument("--is_program_as_sequence", type=int, default=0,
                    help="generate the sequencial program or seperate ops and argus.")

def add_train_relevant_args(parser: ArgumentParser):
    parser.add_argument("--t_bsz", type=int,
                        help="the batch size for training.")
    parser.add_argument("--e_bsz", type=int,
                        help="the batch size for evaluation.")
    
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        help="the accumulation steps to backpropagate.")
    parser.add_argument("--max_epoch", type=int,
                        help="the maximize epoch number to train.")

    parser.add_argument("--lr", type=float,
                        help="learning rate for training the model.")
    parser.add_argument("--weight_decay", type=float,
                        help="weight decay rate.")
    parser.add_argument("--fine_tune", type=int,
                        help="whether fine tune the PLM model.")
    parser.add_argument("--sheduled_sampling", type=int,
                        help="whether enable scheduled sampling.")
    parser.add_argument("--sampling_k", type=int,
                        help="the k for the samling function.")

def add_assist_args(parser: ArgumentParser):
    parser.add_argument("--data_limit", type=int, default=-1,
                        help="limit the number of the data to be processed, used for DEBUG.")

    parser.add_argument("--cuda", type=bool,
                        help="whether use GPU or CPU.")

    parser.add_argument("--seed", type=int, default=17,
                        help="the seed number for the random.")
    
    parser.add_argument("--wandb",
                        help="whether use wandb to record the training.")

    parser.add_argument("--log_per_updates", type=int,
                        help="interval to display the updating information.")

    parser.add_argument("--save_every_steps", type=int,
                        help="interval to save the model.")

def add_test_relevant_args(parser: ArgumentParser):
    parser.add_argument("--e_bsz", type=int,
                        help="batch size for testing data.")
    parser.add_argument("--inference_results_path", type=str,
                        help="the path for saving the inference results.")
    parser.add_argument("--inference_wrong_results_path", type=str,
                        help="the path for saving the wrong inference results.")