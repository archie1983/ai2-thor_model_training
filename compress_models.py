from model_training import HabitatNeuralNetwork, HabitatDataLoading, HabitatNNTrainer
from model_compression_toolkit import CoreConfig, QuantizationConfig, DefaultDict, ptq
from model_compression_toolkit.exporter import pytorch_export_model

class ModelCompressor():
    def __init__(self):
        # construct hyperparameters with the correct architecture
        self.hp = HyperParameters(architecture_id = 2,
                             batch_size = 1, # batch size MUST be 1 because we will use the DataLoader wrapped in a generator function and we will need data one by one
                             data_split = [0.9, 0.1],
                             min_expl_size = 0,
                             use_front_view_only = True)

        # create the model
        #self.model = HabitatNeuralNetwork(hp)
        # Load the model from a saved epoch instead of creating here
        hnt = HabitatNNTrainer(self.hp, load_saved = True, pth_path = 'epoch_0.pth')

        # put it in eval mode. This now is our full, un-compressed model
        self.full_model = hnt.model.eval()

        # Load the data and split it in batches and training and test portions
        dl = HabitatDataLoading(hp)
        (self.train_data_loader, self.test_data_loader) = dl.get_train_test_loaders()

        # Configure quantization
        quant_config = QuantizationConfig(
            activation_n_bits=8,          # Quantize activations to 8-bit
            weights_n_bits=8,             # Quantize weights to 8-bit
            weights_per_channel_threshold=True,  # Per-channel quantization
            enable_weights_quantization=True,
            enable_activation_quantization=True,
        )

        core_config = CoreConfig(
            quantization_config=quant_config,
            # Add pruning config here if needed
        )

        # Compress the model
        self.quantized_model, _ = ptq.pytorch_post_training_quantization_experimental(
            model=self.full_model,
            representative_data_gen=self.representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=DefaultDict()  # Use default target platform
        )

        # Export to ONNX
        pytorch_export_model(
            model=self.quantized_model,
            save_model_path="quantized_model.onnx",
            repr_dataset=self.representative_data_gen,
            target_platform=DefaultDict(),
        )

        self.validate_compressed_model()

    def representative_data_gen(self):
        max_samples = 600
        cnt = 0
        # Iterate through the DataLoader
        for images, actions, path_lengths in self.train_data_loader:
            # print(images, actions, path_lengths)
            #print(actions, path_lengths)
            # put out data on the GPU
            #images = images.to(self.device)
            #actions = actions.to(self.device)

            # Extract input(s) from the batch (ignore labels if they exist)
            # Convert PyTorch tensor to numpy array and yield
            yield [images.numpy().astype(np.float32)]

            # Break when required number of samples has been provided
            cnt += 1
            if cnt > max_samples: break

    def validate_compressed_model(self):
        # Example validation loop
        with torch.no_grad():
            max_samples = 10
            cnt = 0
            # Iterate through the DataLoader
            for images, actions, path_lengths in self.train_data_loader:
                original_output = self.full_model(images)
                quantized_output = self.quantized_model(images)
                print("Original vs. Quantized MSE:", ((original_output - quantized_output)**2).mean())

                # Break when required number of samples has been provided
                cnt += 1
                if cnt > max_samples: break
