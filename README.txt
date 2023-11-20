PACKAGES/MODULES:
    1. utilities (package)
        1.1. general (module)
            General useful utilities for the whole project.
            A module contains a single function get_files_full_paths_list.
        1.2. parser (module)
            A module responsible for defining and parsing execution arguments.
            The execution arguments can be provided through a command line or ".ini" files.
        1.3. logger (module)
            A module responsible for the log messages.
            Provides an easy way to write the log messages to both console and log file/s.
        1.4. audio_features (module)
            This module provides a capability of extracting different useful audio features, e.g., waveforms, spectrograms, MFCCs etc.
            In addition to a capability of the features extraction, it provides methods for the visualization of these features.
        1.5. data_augmentation (module)
            This module implements a technique of the audio data augmentation, based on 5 transformations - AddGaussianNoise, PitchShift, TimeStretch, GainTransition, RoomSimulator.
    2. preprocessing (package)
        2.1. preprocessor (module)
            This module implements a general audio data preprocessing functionality (mostly consists of audio features extraction and data augmentation).
            The main purpose of the module is to prepare the numpy tensors (stored as ".npz" files) from the raw dataset/s, that can be used directly in the models.
            Also serves as an interface (abstract class) for the concrete preprocessors for each dataset:
                2.1.1. crema_d_preprocessor (module)
                    Implements the preprocessing details relevant to the CREMA-D dataset.
                2.1.2. emo_db_preprocessor (module)
                    Implements the preprocessing details relevant to the EMO DB dataset.
                2.1.3. emovo_preprocessor (module)
                    Implements the preprocessing details relevant to the EMOVO dataset.
                2.1.4. ravdess_preprocessor (module)
                    Implements the preprocessing details relevant to the RAVDESS dataset.
                2.1.5. shemo_preprocessor (module)
                    Implements the preprocessing details relevant to the SHEMO dataset.
    3. models (package)
        3.1. model (module)
            Provides a general functionality for any DL model.
            For example - training, evaluation, ability to save and load the model.
            Serves as an interface (abstract class) - the inheritors must implement the "_build_model" method, i.e. to tell how their model should be built.
        3.2. classifier (module)
            Extends the general DL model functionality described above with a general classification models functionality - the "predict" method that allows to make predictions with a trained model.
        3.3. emotion_classifier (module)
            Implements the CNN-MLP model to solve the Speech Emotion Recognition task, according to the architecture described in the project paper.
            Uses (implements) the interface provided by the "model" and the "classifier" modules.
    4. pipelines (package)
        4.1. emotions_classification (module)
            Implements the flow used for the research described in the project paper.
            Provides a functionality for performing a training, evaluation, tuning and K-Fold Cross Validation.



FLOWS (HOW TO RUN):
    0. Help
        Run "main.py -h" to see the help message describing in details all the flags and how to use them.
    1. Preprocessing
        1.1. Create (or modify the existing one) an ".ini" file with the relevant flags for the preprocessing.
            1.1.1. For the supported flags and how to use them, see "utilities/parser.py" and "config/emotion_classification_preprocessing.ini" (as an example).
            1.1.2. The most important flags to set: "preprocess_data = true" (and "augment_data = true" if you'd like to use the augmentation as well).
        1.2. Execute the "main.py" script using the flag "-c <path_to_your_ini_file>", e.g., "python3 main.py -c config/emotion_classification_preprocessing.ini"
        1.3. If "emotion=EMOTION", "dataset=DATASET", "preprocessing_feature=FEATURE", so the preprocessed tensors should be stored at
             "data/EMOTION/preprocessed/DATASET/FEATURE/features.npz" and "data/EMOTION/preprocessed/DATASET/FEATURE/labels.npz".
             For example: "data/angry/preprocessed/emovo/mfcc/features.npz".
    2. Tuning
        2.1. Create (or modify the existing one) an ".ini" file with the relevant flags for the tuning.
            2.1.1. For the supported flags and how to use them, see "utilities/parser.py" and "config/emotion_classification_tuning.ini" (as an example).
            2.1.2. The most important flag to set: "mode = tuning"
        2.2. Open the "pipelines/emotions_classification.py" file, and set the values to tune (for each hyper-parameter) in the "tune" method (lines 176-185).
            2.2.1 Relevant hyper-parameters - channels, mlp_units, pooling_indices, pooling_size, kernel_size, dropout, regularization_coefficient, batch_size, learning_rate.
        2.3. Execute the "main.py" script using the flag "-c <path_to_your_ini_file>", e.g., "python3 main.py -c config/emotion_classification_tuning.ini"
    3. K-Fold CV
        Very similar to the usage in the tuning mode, the only three differences:
        3.1. The example ".ini" file is "config/emotion_classification_cross_validation.ini".
        3.2. In the ".ini" file use "mode = cross_validation".
        3.3. The hyper-parameters values should be changed in the "cross_validation" method (lines 231-238).
    4. Training & Evaluation
        4.1. Create (or modify the existing one) an ".ini" file with the relevant flags for the training/evaluation.
            4.1.1. For the supported flags and how to use them, see "utilities/parser.py", "config/emotion_classification_training.ini" and "config/emotion_classification_evaluation.ini" (as examples).
            4.1.2. The most important flag to set: "mode = training" (or "mode = evaluation)".
            4.1.3. Set the model hyper-parameters using the relevant flags (e.g., "learning_rate", "channels").
        4.2. Use the "save_model_to" flag in the "training" mode to provide a path (without the extension) to where the model should be stored after the training with the specified configuration.
             After the training, the tensors related to the training, validation and test sets will be stored in the same directory with the model.
        4.3. Execute the "main.py" script using the flag "-c <path_to_your_ini_file>", e.g., "python3 main.py -c config/emotion_classification_training.ini"
        4.3. Use the "load_model_from" flag in the "evaluation" mode to specify (without the extension) which model should be loaded for the evaluation.
             The training, validation and test sets will be taken from the directory where the model is located.
        4.4. Execute the "main.py" script using the flag "-c <path_to_your_ini_file>", e.g., "python3 main.py -c config/emotion_classification_evaluation.ini"