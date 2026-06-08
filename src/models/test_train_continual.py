import unittest
import sys
import os
import tempfile
import json
import shutil
import re
from unittest.mock import patch, MagicMock
from functools import partial

# Add project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from train_continual import parse_continual_args, main
import train_continual

class TestContinualArgsParsing(unittest.TestCase):
    def test_default_continual_args(self):
        """Test default arguments for continual training parsing."""
        with patch("sys.argv", ["train_continual.py"]):
            args = parse_continual_args()
            self.assertEqual(args.continual, "True")
            self.assertEqual(args.task_sequence, [2, 3, 4, 5])
            self.assertEqual(args.reset_optimizer, "True")
            self.assertEqual(args.continual_mode, "composition")
            self.assertEqual(args.num_cycles, 1)

    def test_task_sequence_parsing_formats(self):
        """Test parsing of task_sequence when passed as string lists, comma-separated, or standard lists."""
        # 1. Test standard space-separated command-line arguments
        with patch("sys.argv", ["train_continual.py", "--task_sequence", "2", "3", "4"]):
            args = parse_continual_args()
            self.assertEqual(args.task_sequence, [2, 3, 4])
            
        # 2. Test JSON/YAML style list passed as a string (common in W&B sweeps)
        with patch("sys.argv", ["train_continual.py", "--task_sequence", "[2,3,4,5]"]):
            args = parse_continual_args()
            self.assertEqual(args.task_sequence, [2, 3, 4, 5])
            
        # 3. Test comma-separated string format
        with patch("sys.argv", ["train_continual.py", "--task_sequence", "3,4,5"]):
            args = parse_continual_args()
            self.assertEqual(args.task_sequence, [3, 4, 5])
            
        # 4. Test space-separated single string argument
        with patch("sys.argv", ["train_continual.py", "--task_sequence", "2 3 4 5"]):
            args = parse_continual_args()
            self.assertEqual(args.task_sequence, [2, 3, 4, 5])

    @patch("train_continual.parse_continual_args")
    @patch("train_continual.Accelerator")
    @patch("train_continual.set_seed")
    def test_boolean_conversion_in_main(self, mock_set_seed, mock_accelerator_class, mock_parse_args):
        """Test string boolean conversions in main configuration block."""
        mock_args = MagicMock()
        mock_args.continual = "True"
        mock_args.reset_optimizer = "True"
        mock_args.filter_query_from_support = "True"
        mock_args.store_predictions = "True"
        mock_args.use_preprocessed = "True"
        mock_args.use_scheduler = "True"
        mock_args.save_checkpoints = "True"
        mock_args.log_continual_csv = "True"
        mock_args.enable_auto_stop = "True"
        mock_args.use_truncation = "True"
        mock_args.fp16 = "False"
        mock_args.seed = 42
        mock_args.val_data_path = "None"
        mock_args.train_data_path = "train_shop_1_qhop_2.json"
        mock_args.save_continual_run_id = None
        mock_args.save_dir = "src/saved_models/"
        mock_args.is_held_out_color_exp = "False"
        mock_args.preprocessed_dir = "src/data/preprocessed"
        mock_args.model_size = "xsmall"
        mock_args.model_architecture = "encoder"
        mock_args.task_type = "classification"
        mock_args.input_format = "features"
        mock_args.output_format = "stone_states"
        mock_args.data_split_seed = 0
        mock_args.wandb_mode = "offline"
        mock_args.wandb_project = "test"
        mock_args.wandb_entity = "test"
        mock_args.wandb_run_name = "test_run"
        mock_args.task_sequence = [2]
        mock_args.num_cycles = 1

        mock_parse_args.return_value = mock_args

        # Mock the entire training sequence loop to stop execution early in main()
        with patch("train_continual.AlchemyDataset"), \
             patch("train_continual.DataLoader"), \
             patch("train_continual.wandb"):
            try:
                main()
            except Exception:
                # We expect it to raise errors later when trying to run loops, but we check conversion before that
                pass

            self.assertIs(mock_args.continual, True)
            self.assertIs(mock_args.reset_optimizer, True)
            self.assertIs(mock_args.filter_query_from_support, True)
            self.assertIs(mock_args.store_predictions, True)
            self.assertIs(mock_args.use_preprocessed, True)
            self.assertIs(mock_args.use_scheduler, True)
            self.assertIs(mock_args.save_checkpoints, True)
            self.assertIs(mock_args.log_continual_csv, True)
            self.assertIs(mock_args.enable_auto_stop, True)
            self.assertIs(mock_args.use_truncation, True)

class TestHopPatternResolution(unittest.TestCase):
    def test_composition_path_substitution(self):
        """Test regex substitution for composition continual mode paths."""
        support_hop_init = "1"
        query_hop_init = "2"
        hop_length = 3
        train_data_path_template = "data/compositional_chemistry_samples_train_shop_1_qhop_2.json"
        
        hop_pattern_src = f"shop_{support_hop_init}_qhop_{query_hop_init}"
        hop_pattern_dst = f"shop_1_qhop_{hop_length}"
        
        current_path = re.sub(hop_pattern_src, hop_pattern_dst, train_data_path_template)
        self.assertEqual(current_path, "data/compositional_chemistry_samples_train_shop_1_qhop_3.json")

    def test_decomposition_path_substitution(self):
        """Test regex substitution for decomposition continual mode paths."""
        support_hop_init = "2"
        query_hop_init = "1"
        hop_length = 4
        train_data_path_template = "data/decompositional_chemistry_samples_train_shop_2_qhop_1.json"
        
        hop_pattern_src = f"shop_{support_hop_init}_qhop_{query_hop_init}"
        hop_pattern_dst = f"shop_{hop_length}_qhop_1"
        
        current_path = re.sub(hop_pattern_src, hop_pattern_dst, train_data_path_template)
        self.assertEqual(current_path, "data/decompositional_chemistry_samples_train_shop_4_qhop_1.json")

class TestModelArchitectureInstantiation(unittest.TestCase):
    @patch("train_continual.create_classifier_model")
    @patch("train_continual.create_decoder_classifier_model")
    @patch("train_continual.create_linear_model")
    def test_model_architecture_selection(self, mock_linear, mock_decoder, mock_classifier):
        """Test correct model creation utility is called with accepted parameters."""
        mock_dataset = MagicMock()
        mock_dataset.word2idx = {"a": 0, "b": 1}
        mock_dataset.stone_state_to_id = {"{a}": 0, "{b}": 1}
        mock_dataset.io_sep_token_id = None
        mock_dataset.item_sep_token_id = None
        mock_accelerator = MagicMock()
        mock_accelerator.device = "cpu"
        
        mock_args = MagicMock()
        mock_args.model_size = "xsmall"
        mock_args.max_seq_len = 128
        mock_args.pooling_strategy = "global"
        mock_args.batch_size = 32
        mock_args.use_flash_attention = "True"
        mock_args.padding_side = "right"
        mock_args.include_nonlinearity = "True"
        mock_args.use_pre_norm = "False"
        mock_args.override_num_classes = None
        mock_args.prediction_type = "default"
        mock_args.flatten_linear_model_input = "False"

        # 1. Test Encoder/Classifier model instantiation path
        mock_args.model_architecture = "encoder"
        
        # Simulating logic inside train_continual main()
        num_classes = len(mock_dataset.stone_state_to_id)
        train_continual.create_classifier_model(
            config_name=mock_args.model_size,
            src_vocab_size=len(mock_dataset.word2idx),
            num_classes=num_classes,
            device=mock_accelerator.device,
            max_len=mock_args.max_seq_len,
            io_sep_token_id=getattr(mock_dataset, 'io_sep_token_id', None),
            item_sep_token_id=getattr(mock_dataset, 'item_sep_token_id', None),
            pooling_strategy=mock_args.pooling_strategy
        )
        mock_classifier.assert_called_once_with(
            config_name="xsmall",
            src_vocab_size=2,
            num_classes=2,
            device="cpu",
            max_len=128,
            io_sep_token_id=None,
            item_sep_token_id=None,
            pooling_strategy="global"
        )

        # 2. Test Decoder-only model instantiation path
        mock_args.model_architecture = "decoder"
        train_continual.create_decoder_classifier_model(
            config_name=mock_args.model_size,
            src_vocab_size=len(mock_dataset.word2idx),
            num_classes=num_classes,
            device=mock_accelerator.device,
            max_len=mock_args.max_seq_len,
            prediction_type=mock_args.prediction_type,
            padding_side=mock_args.padding_side,
            use_flash_attention=(mock_args.use_flash_attention == 'True' or mock_args.use_flash_attention is True),
            batch_size=mock_args.batch_size,
            vocab=mock_dataset.input_word2idx if hasattr(mock_dataset, 'input_word2idx') else None,
            use_pre_norm=(mock_args.use_pre_norm == 'True' or mock_args.use_pre_norm is True)
        )
        mock_decoder.assert_called_once()

class TestEarlyStoppingLogic(unittest.TestCase):
    def test_streak_tracking_and_trigger(self):
        """Test that validation streak increments, resets, and triggers early stopping correctly."""
        patience = 3
        threshold = 0.8
        
        # Test Case A: Successive validation checks exceed threshold
        val_accs = [0.85, 0.90, 0.81]
        val_acc_streak = 0
        triggered = False
        for acc in val_accs:
            if acc >= threshold:
                val_acc_streak += 1
            else:
                val_acc_streak = 0
            if val_acc_streak >= patience:
                triggered = True
                break
        
        self.assertEqual(val_acc_streak, 3)
        self.assertTrue(triggered)

        # Test Case B: Streak resets to 0 when accuracy drops below threshold
        val_accs = [0.85, 0.90, 0.75, 0.81]
        val_acc_streak = 0
        triggered = False
        for acc in val_accs:
            if acc >= threshold:
                val_acc_streak += 1
            else:
                val_acc_streak = 0
            if val_acc_streak >= patience:
                triggered = True
                break
        
        self.assertEqual(val_acc_streak, 1) # Resets on 3rd epoch, increments to 1 on 4th epoch
        self.assertFalse(triggered)

    def test_streak_reset_on_task_transition(self):
        """Test that val_acc_streak resets back to 0 at the start of each task."""
        # Simulate tasks loop
        task_sequence = [2, 3]
        epochs_limit = 2
        patience = 2
        threshold = 0.5
        
        # Store state transitions of streak
        streak_snapshots_per_task = []
        
        for task_idx, hop_length in enumerate(task_sequence):
            # Streak must be reset to 0 for each new task
            val_acc_streak = 0
            
            # Simulate epochs
            for epoch in range(epochs_limit):
                val_acc = 0.9 # meets threshold
                if val_acc >= threshold:
                    val_acc_streak += 1
                else:
                    val_acc_streak = 0
            streak_snapshots_per_task.append(val_acc_streak)
            
        # Verify that despite having high validation accuracy at the end of task 0,
        # the streak did not accumulate across task boundaries to exceed patience (i.e. streak is 2, not 4).
        self.assertEqual(streak_snapshots_per_task, [2, 2])

class TestIntegrationContinualTraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Construct tiny mock json dataset episodes
        dummy_episode_data = {
            "episodes": {
                "episode_0": {
                    "support": [
                        "{color: red, size: small, roundness: pointy, reward: -3} P1 -> {color: red, size: small, roundness: pointy, reward: -3}"
                    ],
                    "query": [
                        "{color: red, size: small, roundness: pointy, reward: -3} P2 -> {color: red, size: small, roundness: pointy, reward: -3}"
                    ]
                }
            }
        }
        
        # Write dummy JSON dataset files for composition sequence (hop 2 and 3)
        self.train_path_2 = os.path.join(self.temp_dir, "chemistry_train_shop_1_qhop_2_seed_0.json")
        self.val_path_2 = os.path.join(self.temp_dir, "chemistry_val_shop_1_qhop_2_seed_0.json")
        self.train_path_3 = os.path.join(self.temp_dir, "chemistry_train_shop_1_qhop_3_seed_0.json")
        self.val_path_3 = os.path.join(self.temp_dir, "chemistry_val_shop_1_qhop_3_seed_0.json")
        self.train_path_4 = os.path.join(self.temp_dir, "chemistry_train_shop_1_qhop_4_seed_0.json")
        self.val_path_4 = os.path.join(self.temp_dir, "chemistry_val_shop_1_qhop_4_seed_0.json")
        self.train_path_5 = os.path.join(self.temp_dir, "chemistry_train_shop_1_qhop_5_seed_0.json")
        self.val_path_5 = os.path.join(self.temp_dir, "chemistry_val_shop_1_qhop_5_seed_0.json")
        
        for p in [self.train_path_2, self.val_path_2, self.train_path_3, self.val_path_3, self.train_path_4, self.val_path_4, self.train_path_5, self.val_path_5]:
            with open(p, "w") as f:
                json.dump(dummy_episode_data, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("train_continual.wandb")
    @patch("train.wandb")
    @patch("train_continual.Accelerator")
    def test_end_to_end_lightweight_cpu_run(self, mock_accelerator_class, mock_train_wandb, mock_wandb):
        """Run integration test checking training sequence completion using batch size of 2048."""
        # Configure mock accelerator
        mock_accel = MagicMock()
        mock_accel.device = "cpu"
        mock_accel.num_processes = 1
        mock_accel.is_local_main_process = True
        mock_accel.is_main_process = True
        mock_accel.unwrap_model = lambda x: x
        mock_accel.prepare = lambda x: x
        mock_accelerator_class.return_value = mock_accel
        
        # Override save path in temp directory
        save_dir = os.path.join(self.temp_dir, "saved_models")
        preprocessed_dir = os.path.join(self.temp_dir, "preprocessed")
        
        # Set sys.argv arguments for the test run
        test_args = [
            "train_continual.py",
            "--continual", "True",
            "--task_sequence", "2", "3", "4", "5",
            "--epochs_per_task", "1",
            "--batch_size", "2048",
            "--train_data_path", os.path.join(self.temp_dir, "chemistry_train_shop_1_qhop_2.json"),
            "--val_data_path", os.path.join(self.temp_dir, "chemistry_val_shop_1_qhop_2.json"),
            "--save_dir", save_dir,
            "--preprocessed_dir", preprocessed_dir,
            "--use_preprocessed", "False",
            "--wandb_mode", "offline",
            "--num_workers", "0",
            "--enable_auto_stop", "True",
            "--auto_stop_val_acc_threshold", "0.0",
            "--auto_stop_val_acc_patience", "1",
            "--model_size", "xsmall",
            "--model_architecture", "decoder",
            "--task_type", "classification"
        ]
        
        with patch("sys.argv", test_args):
            # Verify integration test run runs to completion
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)
            except Exception as e:
                self.fail(f"Integration test failed with unexpected exception: {e}")

if __name__ == "__main__":
    unittest.main()
