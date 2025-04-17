# BioML Models Dashboard

A Streamlit dashboard for interacting with various bio ML models deployed on the cloud.

## Models Included

- **Boltz**: Protein structure prediction
- **ColabFold**: Protein structure prediction with AMBER refinement
- **ESMFold**: Protein structure prediction and inverse folding
- **ProGPT2**: Protein sequence generation
- **ProteinMPNN**: Protein design
- **RFDiffusion**: Protein design with diffusion models

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```
   streamlit run main.py
   ```

3. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Usage

1. Select a model from the dropdown menu
2. Fill in the required parameters for the selected model
3. Click the "Run Model" button to execute the model
4. View the results in the dashboard

## File Upload Functionality

For models that require S3 URIs (ESMFold, ProteinMPNN, and RFDiffusion), you can upload local files directly:

1. Provide your AWS credentials (Access Key ID and Secret Access Key)
2. Use the file upload button to select a local file
3. The file will be automatically uploaded to S3 and the S3 URI will be filled in
4. Complete the remaining parameters and run the model

## Notes

- For models that require AWS credentials, you'll need to provide your AWS access key ID and secret access key
- Some models may take a while to run, especially for complex inputs
- The default values provided are examples and may need to be adjusted based on your specific use case
- Files uploaded to S3 are stored in the "inference-files-s3" bucket in the "input" folder with a unique UUID # streamlit-bioml
