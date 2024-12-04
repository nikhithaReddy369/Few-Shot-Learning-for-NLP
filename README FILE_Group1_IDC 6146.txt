## Usage

The notebook is divided into two main parts:

### Part 1: Question Answering on SQuAD

This section evaluates multiple models on the SQuAD dataset to perform question-answering tasks.

#### 1. Run the Question Answering Cells

*Execute the Cells Sequentially*

Start from importing libraries to loading models and processing the dataset.

*Included Cells:*

Importing necessary libraries.
Loading and preprocessing the SQuAD dataset.
Defining helper functions for evaluation.
Loading and initializing models: BLOOM, BERT, RoBERTa, and OpenAI GPT.
Applying few-shot learning techniques with predefined examples.
Generating answers for each question in the dataset using each model.
Calculating F1 Scores to evaluate the accuracy of the answers.
Saving the results to a CSV file and generating summary plots.

#### 2. Understanding the Workflow

*Data Loading:* The SQuAD dataset is loaded and a subset is selected for evaluation.
*Model Initialization:* Models are loaded with appropriate configurations, leveraging GPU acceleration.
*Few-Shot Learning:* Predefined examples are used to guide the models in generating answers.
*Evaluation:* F1 Scores are computed to measure the performance of each model.
*Visualization:* Results are saved and visualized through plots for easy comparison.

### Part 2: Translation on WMT14

This section evaluates various translation models on the WMT14 dataset to perform English-to-French translations.

#### 1. Run the Translation Cells

*Execute the Cells Sequentially*

Start from setting up the device to loading models and processing the dataset.

*Included Cells:*

Importing additional libraries.
Setting up the device (GPU).
Loading the WMT14 French-English dataset.
Defining few-shot examples for translation tasks.
Loading and initializing translation models: BLOOM, MarianMT, T5, and OpenAI GPT.
Performing translations using each model with different few-shot settings.
Calculating BLEU Scores to evaluate the quality of translations.
Saving the results to a CSV file and generating summary plots.

#### 2. Understanding the Workflow

*Data Loading:* The WMT14 dataset is loaded and a subset is selected for evaluation.
*Model Initialization:* Translation models are loaded with appropriate configurations, leveraging GPU acceleration.
*Few-Shot Learning:* Predefined examples are used to guide the models in generating translations.
*Evaluation:* BLEU Scores are computed to measure the performance of each model.
*Visualization:* Results are saved and visualized through plots for easy comparison.

## Results

After running both parts of the evaluation, the following outputs are generated:

### CSV Files

squad_results_few_shot_comparison.csv: Contains the answers and F1 Scores for each model on the SQuAD dataset.
translation_bleu_scores.csv: Contains the BLEU Scores for each translation model on the WMT14 dataset.

### Plots

*Question Answering Models:*
  Bar charts comparing the *Average F1 Scores* of the Question Answering models.
*Translation Models:*
  Bar charts comparing the *BLEU Scores* of the Translation models.

### Summary Tables

*Printed in the Notebook's Output Cells:*
  Provides a quick overview of model performances.

## Troubleshooting

### GPU Not Detected

*Ensure that the Runtime is Set to Use a GPU:*
  
Runtime > Change runtime type > Hardware accelerator: GPU

markdown
Copy code

*Verify GPU Availability in the Notebook:*

import torch print(torch.cuda.is_available()) print(torch.cuda.device_count()) print(torch.cuda.get_device_name(0))

markdown
Copy code

### Memory Issues

Google Colab provides limited GPU memory. If you encounter out-of-memory errors:

*Use Smaller Models:*

For example, use t5-small instead of larger variants.

*Reduce the Batch Size When Processing Data.*

*Clear GPU Cache:*

import torch torch.cuda.empty_cache()

markdown
Copy code

### API Errors

*OpenAI API:*

Ensure your API key is correct and has sufficient permissions.
Check your OpenAI account for usage limits.

*Hugging Face:*

Verify that your token is correct and has access to the required models.

### Package Compatibility

*Ensure All Packages Are Up to Date. Reinstall or upgrade if necessary:*

!pip install --upgrade transformers datasets sacrebleu sentencepiece accelerate evaluate openai

markdown
Copy code

###Runtime Disconnections

Google Colab has usage limits. If your session disconnects:

*Save Your Progress Frequently.*
*Consider Splitting Tasks into Smaller Chunks.*

## License

This project is licensed under the MIT License.

## Additional Notes

*Security Considerations*

*API Keys:* Avoid hardcoding API keys in the notebook. Use environment variables or Colab's secret management to securely handle sensitive information.
*Public Sharing:* If sharing the notebook publicly, ensure that all sensitive information (like API keys) is removed or secured.

*Performance Optimization*

Utilize batch processing where possible to leverage GPU parallelism.
Monitor GPU usage to ensure efficient resource utilization.

*Extensibility*

The notebook is modular, allowing you to add or replace models as needed.
You can extend the evaluation metrics or incorporate additional datasets for more comprehensive analysis.

#### Contact

For any questions or suggestions, please contact

eb128@students.Uwf.edu
ma263@students.uwf.edu
rc140@students.uwf.edu
na104@students.uwf.edu

## Acknowledgements

[Hugging Face](https://huggingface.co/) for providing a vast repository of NLP models.
[OpenAI](https://openai.com/) for their powerful GPT models.
[Google Colab](https://colab.research.google.com/) for facilitating GPU-accelerated computations.
The authors and contributors of the SQuAD and WMT14 datasets.