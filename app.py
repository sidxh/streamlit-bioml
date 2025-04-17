import streamlit as st
import requests
import json
import os
import boto3
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time

# Set page config
st.set_page_config(
    page_title="BioML Models Dashboard",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'model_runs' not in st.session_state:
    st.session_state.model_runs = {}
if 'last_run_id' not in st.session_state:
    st.session_state.last_run_id = None

# Define model endpoints
MODEL_ENDPOINTS = {
    "Boltz": "http://a914f8928ab3d48e0b8e7b0deb6ae5c5-1777632126.us-east-1.elb.amazonaws.com/svc/default/boltz-serverless-gpus-1-t4/invocations",
    "ColabFold": "http://a914f8928ab3d48e0b8e7b0deb6ae5c5-1777632126.us-east-1.elb.amazonaws.com/svc/default/collabfold-serverless-gpus-1-t4/invocations",
    "ESMFold": "http://a914f8928ab3d48e0b8e7b0deb6ae5c5-1777632126.us-east-1.elb.amazonaws.com/svc/default/esmfold-serverless-gpus-1-a10g/invocations",
    "ProGPT2": "http://a914f8928ab3d48e0b8e7b0deb6ae5c5-1777632126.us-east-1.elb.amazonaws.com/svc/default/progpt2-serverless-gpus-1-t4/invocations",
    "ProteinMPNN": "http://a914f8928ab3d48e0b8e7b0deb6ae5c5-1777632126.us-east-1.elb.amazonaws.com/svc/default/proteinmpnn-serverless-gpus-1-t4/invocations",
    "RFDiffusion": "http://a914f8928ab3d48e0b8e7b0deb6ae5c5-1777632126.us-east-1.elb.amazonaws.com/svc/default/rfdiffusion-serverless-gpus-1-t4/invocations"
}

# Define model parameters
MODEL_PARAMS = {
    "Boltz": {
        "s3_uri": {"type": "text_input", "label": "S3 URI", "default": "s3://inference-files-s3/input/boltz-serverless/input.fasta", "file_upload": True},
        "aws_access_key_id": {"type": "text_input", "label": "AWS Access Key ID", "default": ""},
        "aws_secret_access_key": {"type": "text_input", "label": "AWS Secret Access Key", "default": ""},
        "aws_region": {"type": "text_input", "label": "AWS Region", "default": "us-east-1"}
    },
    "ColabFold": {
        "sequence": {"type": "text_area", "label": "Protein Sequence", "default": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"},
        "name": {"type": "text_input", "label": "Protein Name", "default": "test_protein"},
        "amber": {"type": "checkbox", "label": "Use AMBER", "default": True},
        "num_recycles": {"type": "number_input", "label": "Number of Recycles", "default": 3, "min": 1, "max": 10},
        "aws_access_key_id": {"type": "text_input", "label": "AWS Access Key ID", "default": ""},
        "aws_secret_access_key": {"type": "text_input", "label": "AWS Secret Access Key", "default": ""},
        "aws_region": {"type": "text_input", "label": "AWS Region", "default": "us-east-1"},
        "s3_bucket": {"type": "text_input", "label": "S3 Bucket", "default": "inference-files-s3"}
    },
    "ESMFold": {
        "task_type": {"type": "selectbox", "label": "Task Type", "options": ["fold", "inverse_fold"], "default": "fold"},
        "sequence": {"type": "text_area", "label": "Protein Sequence", "default": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "condition": "task_type == 'fold'"},
        "s3_uri": {"type": "text_input", "label": "S3 URI", "default": "s3://inference-files-s3/input/5yh2_c_chain.pdb", "condition": "task_type == 'inverse_fold'", "file_upload": True},
        "aws_access_key_id": {"type": "text_input", "label": "AWS Access Key ID", "default": "", "condition": "task_type == 'inverse_fold'"},
        "aws_secret_access_key": {"type": "text_input", "label": "AWS Secret Access Key", "default": "", "condition": "task_type == 'inverse_fold'"},
        "aws_region": {"type": "text_input", "label": "AWS Region", "default": "us-east-1", "condition": "task_type == 'inverse_fold'"},
        "chain_id": {"type": "text_input", "label": "Chain ID", "default": "C", "condition": "task_type == 'inverse_fold'"}
    },
    "ProGPT2": {
        "sequence": {"type": "text_input", "label": "Sequence", "default": "<|endoftext|>"},
        "max_length": {"type": "number_input", "label": "Max Length", "default": 100, "min": 1, "max": 500},
        "num_sequences": {"type": "number_input", "label": "Number of Sequences", "default": 2, "min": 1, "max": 10},
        "do_sample": {"type": "checkbox", "label": "Do Sample", "default": True},
        "top_k": {"type": "number_input", "label": "Top K", "default": 950, "min": 1, "max": 1000},
        "repetition_penalty": {"type": "number_input", "label": "Repetition Penalty", "default": 1.2, "min": 1.0, "max": 2.0, "step": 0.1},
        "temperature": {"type": "number_input", "label": "Temperature", "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1},
        "aws_access_key_id": {"type": "text_input", "label": "AWS Access Key ID", "default": ""},
        "aws_secret_access_key": {"type": "text_input", "label": "AWS Secret Access Key", "default": ""},
        "aws_region": {"type": "text_input", "label": "AWS Region", "default": "us-east-1"}
    },
    "ProteinMPNN": {
        "aws_access_key_id": {"type": "text_input", "label": "AWS Access Key ID", "default": ""},
        "aws_secret_access_key": {"type": "text_input", "label": "AWS Secret Access Key", "default": ""},
        "aws_region": {"type": "text_input", "label": "AWS Region", "default": "us-east-1"},
        "s3_uris": {"type": "text_input", "label": "S3 URIs (comma-separated)", "default": "s3://inference-files-s3/input/test.pdb", "file_upload": True},
        "num_seq_per_target": {"type": "number_input", "label": "Number of Sequences per Target", "default": 2, "min": 1, "max": 10},
        "sampling_temp": {"type": "text_input", "label": "Sampling Temperature", "default": "0.1"},
        "seed": {"type": "number_input", "label": "Seed", "default": 37, "min": 1, "max": 1000},
        "batch_size": {"type": "number_input", "label": "Batch Size", "default": 1, "min": 1, "max": 10}
    },
    "RFDiffusion": {
        "contig": {"type": "text_input", "label": "Contig", "default": "[100-100]"},
        "s3_uri": {"type": "text_input", "label": "S3 URI", "default": "s3://inference-files-s3/input/5TPN.pdb", "file_upload": True},
        "aws_access_key_id": {"type": "text_input", "label": "AWS Access Key ID", "default": ""},
        "aws_secret_access_key": {"type": "text_input", "label": "AWS Secret Access Key", "default": ""},
        "aws_region": {"type": "text_input", "label": "AWS Region", "default": "us-east-1"},
        "num_designs": {"type": "number_input", "label": "Number of Designs", "default": 5, "min": 1, "max": 20}
    }
}

def generate_run_id():
    """Generate a unique run ID with timestamp."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def upload_to_s3(file, aws_credentials: Dict[str, str], prefix: str) -> Optional[str]:
    """Upload a file to S3 and return the S3 URI."""
    try:
        # Extract credentials
        aws_access_key_id = aws_credentials.get("aws_access_key_id")
        aws_secret_access_key = aws_credentials.get("aws_secret_access_key")
        aws_region = aws_credentials.get("aws_region", "us-east-1")
        
        if not all([aws_access_key_id, aws_secret_access_key]):
            st.error("Please provide AWS credentials")
            return None
            
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.name)[1]
        s3_key = f"input/{prefix}/{file.name.split('.')[0]}_{timestamp}{file_extension}"
        
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Upload to S3
        s3_client.put_object(
            Bucket="inference-files-s3",
            Key=s3_key,
            Body=file.getvalue()
        )
        
        s3_uri = f"s3://inference-files-s3/{s3_key}"
        
        # Store in session state
        st.session_state.uploaded_files[prefix] = {
            'uri': s3_uri,
            'timestamp': timestamp,
            'filename': file.name
        }
        
        return s3_uri
        
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return None

def create_aws_credentials_section():
    """Create AWS credentials input section."""
    with st.expander("AWS Credentials (Required)", expanded=True):
        aws_access_key = st.text_input("AWS Access Key ID", key="aws_access_key")
        aws_secret_key = st.text_input("AWS Secret Access Key", type="password", key="aws_secret_key")
        aws_region = st.text_input("AWS Region", value="us-east-1", key="aws_region")
        
        return {
            "aws_access_key_id": aws_access_key,
            "aws_secret_access_key": aws_secret_key,
            "aws_region": aws_region
        }

def handle_file_upload(label: str, file_types: List[str], prefix: str, aws_credentials: Dict[str, str]) -> Optional[str]:
    """Handle file upload with S3 integration."""
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(label, type=file_types, key=f"uploader_{label}")
    
    # Check if we have a stored S3 URI for this prefix
    stored_file = st.session_state.uploaded_files.get(prefix, {})
    s3_uri = stored_file.get('uri')
    
    if uploaded_file:
        with col2:
            if st.button("Upload to S3", key=f"upload_btn_{label}"):
                with st.spinner("Uploading to S3..."):
                    s3_uri = upload_to_s3(uploaded_file, aws_credentials, prefix)
                    if s3_uri:
                        st.success("Upload successful!")
    
    # Always show the S3 URI if we have one (either from current upload or stored)
    if s3_uri:
        with col3:
            st.text_input("S3 URI", value=s3_uri, key=f"s3_uri_{label}", disabled=True)
    
    return s3_uri

def format_duration(start_time_str: str) -> str:
    """Format the duration since start time."""
    start_time = datetime.fromisoformat(start_time_str)
    duration = datetime.now() - start_time
    minutes = int(duration.total_seconds() // 60)
    seconds = int(duration.total_seconds() % 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_boltz_form(aws_credentials: Dict[str, str]):
    """Create Boltz model form."""
    s3_uri = handle_file_upload("Upload FASTA file", ["fasta"], "boltz-serverless", aws_credentials)
    
    # Create a container for the run button and status
    run_container = st.container()
    
    with run_container:
        if s3_uri:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Run Model", key="boltz_file_run"):
                    # Generate a new run ID
                    run_id = generate_run_id()
                    st.session_state.last_run_id = run_id
                    
                    # Store run information
                    st.session_state.model_runs[run_id] = {
                        'model': 'Boltz',
                        'status': 'running',
                        'start_time': datetime.now().isoformat(),
                        'params': {
                            "s3_uri": s3_uri,
                            **aws_credentials
                        }
                    }
                    
                    return st.session_state.model_runs[run_id]['params']
            
            # Show run status if there's an active run
            if st.session_state.last_run_id:
                with col2:
                    run_info = st.session_state.model_runs[st.session_state.last_run_id]
                    if run_info['status'] == 'running':
                        duration = format_duration(run_info['start_time'])
                        st.info(f"⏳ Model is running... (Time elapsed: {duration})")
                        st.empty().info(f"⏳ Model is running... (Time elapsed: {duration})")
    
    return None

def create_progpt2_form(aws_credentials: Dict[str, str]):
    """Create ProGPT2 model form."""
    with st.form("progpt2_form"):
        params = {
            "sequence": st.text_input("Sequence", value="<|endoftext|>"),
            "max_length": st.number_input("Max Length", value=100, min_value=1, max_value=500),
            "num_sequences": st.number_input("Number of Sequences", value=2, min_value=1, max_value=10),
            "do_sample": st.checkbox("Do Sample", value=True),
            "top_k": st.number_input("Top K", value=950, min_value=1, max_value=1000),
            "repetition_penalty": st.number_input("Repetition Penalty", value=1.2, min_value=1.0, max_value=2.0, step=0.1),
            "temperature": st.number_input("Temperature", value=0.8, min_value=0.1, max_value=2.0, step=0.1),
            **aws_credentials
        }
        
        if st.form_submit_button("Run Model"):
            return params
    return None

def create_proteinmpnn_form(aws_credentials: Dict[str, str]):
    """Create ProteinMPNN model form."""
    # File upload section
    st.subheader("Upload PDB File")
    
    # Single file upload
    s3_uri = handle_file_upload("Upload PDB file", ["pdb"], "proteinmpnn-serverless", aws_credentials)
    
    if s3_uri:
        with st.form("proteinmpnn_form"):
            num_seq = st.number_input("Number of Sequences per Target", value=2, min_value=1, max_value=10)
            
            # Create the parameters dict exactly matching the curl command format and order
            params = {
                "aws_access_key_id": aws_credentials.get("aws_access_key_id", ""),
                "aws_secret_access_key": aws_credentials.get("aws_secret_access_key", ""),
                "aws_region": aws_credentials.get("aws_region", "us-east-1"),
                "s3_uris": [s3_uri],  # Make sure it's a list with single item
                "num_seq_per_target": int(num_seq)  # Ensure it's an integer
            }
            
            # Debug info (hidden in collapsed section)
            with st.expander("Debug Info"):
                # Show full parameters with masked sensitive data
                debug_params = {
                    **params,
                    "aws_access_key_id": "***" if params["aws_access_key_id"] else "",
                    "aws_secret_access_key": "***" if params["aws_secret_access_key"] else ""
                }
                st.code(json.dumps(debug_params, indent=2))
            
            if st.form_submit_button("Run Model"):
                # Validate required fields
                if not all([
                    params["aws_access_key_id"],
                    params["aws_secret_access_key"],
                    params["aws_region"],
                    params["s3_uris"],
                    isinstance(params["num_seq_per_target"], int)
                ]):
                    st.error("Missing required parameters. Please check all fields are filled.")
                    return None
                
                # Log the exact structure being sent (excluding sensitive data)
                safe_params = {
                    "aws_region": params["aws_region"],
                    "s3_uris": params["s3_uris"],
                    "num_seq_per_target": params["num_seq_per_target"]
                }
                st.info(f"Sending request with parameters:\n```json\n{json.dumps(safe_params, indent=2)}\n```")
                
                return params
    return None

def parse_proteinmpnn_response(response_text: str) -> Dict[str, Any]:
    """Parse ProteinMPNN response text into a structured format."""
    try:
        # Remove any leading/trailing quotes and handle escaped quotes
        cleaned_text = response_text.strip().strip('"').replace('\\"', '"')
        
        # Parse the JSON
        result = json.loads(cleaned_text)
        
        # Return the raw result if it matches the expected format
        if "" in result and isinstance(result[""], list):
            return result
        
        return {"error": "Unexpected response format"}
        
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse response: {str(e)}"}

def create_rfdiffusion_form(aws_credentials: Dict[str, str]):
    """Create RFDiffusion model form."""
    with st.form("rfdiffusion_form"):
        params = {
            "contig": st.text_input("Contig", value="[100-100]"),
            **aws_credentials
        }
        
        submit = st.form_submit_button("Run Model")
        
    # Handle optional PDB file upload outside form
    s3_uri = handle_file_upload("Upload PDB file (optional)", ["pdb"], "rfdiffusion", aws_credentials)
    if s3_uri:
        params["s3_uri"] = s3_uri
    
    if submit or (s3_uri and st.button("Run Model with Uploaded File", key="rfdiffusion_file_run")):
        return params
    
    return None

def create_colabfold_form(aws_credentials: Dict[str, str]):
    """Create ColabFold model form."""
    s3_uri = handle_file_upload("Upload FASTA file", ["fasta"], "colabfold", aws_credentials)
    
    with st.form("colabfold_form"):
        params = {
            "amber": st.checkbox("Use AMBER", value=True),
            "num_recycles": st.number_input("Number of Recycles", value=3, min_value=1, max_value=10),
            "model_type": st.selectbox("Model Type", ["auto", "alphafold2_ptm", "alphafold2"], key="colabfold_model_type"),
            "use_templates": st.checkbox("Use Templates", value=False),
            **aws_credentials
        }
        
        if s3_uri:
            params["s3_uri"] = s3_uri
        
        if st.form_submit_button("Run Model"):
            return params
    return None

def call_model_api(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call model API endpoint with detailed logging."""
    try:
        # Update run status to show request details
        if st.session_state.last_run_id:
            run_info = st.session_state.model_runs[st.session_state.last_run_id]
            run_info['request_time'] = datetime.now().isoformat()
            
            # Log request details (excluding sensitive info)
            safe_params = {k: v for k, v in params.items() 
                         if k not in ['aws_access_key_id', 'aws_secret_access_key']}
            st.info(f"📤 Sending request to model API...\nEndpoint: {endpoint}\nParameters: {json.dumps(safe_params, indent=2)}")
        
        # For ProteinMPNN, ensure the parameters are exactly as expected
        if "proteinmpnn" in endpoint.lower():
            # Verify all required fields are present
            required_fields = [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_region",
                "s3_uris",
                "num_seq_per_target"
            ]
            
            missing_fields = [field for field in required_fields if field not in params or not params[field]]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Ensure s3_uris is a list
            if not isinstance(params["s3_uris"], list):
                params["s3_uris"] = [params["s3_uris"]]
            
            # Ensure num_seq_per_target is an integer
            params["num_seq_per_target"] = int(params["num_seq_per_target"])
            
            # Create a new dict with exact order matching curl command
            params = {
                "aws_access_key_id": params["aws_access_key_id"],
                "aws_secret_access_key": params["aws_secret_access_key"],
                "aws_region": params["aws_region"],
                "s3_uris": params["s3_uris"],
                "num_seq_per_target": params["num_seq_per_target"]
            }
        
        # Make the API call
        response = requests.post(
            endpoint,
            data=json.dumps(params),  # Explicitly convert to JSON string
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout
        )
        
        # Log the raw response for debugging
        if "proteinmpnn" in endpoint.lower():
            st.info(f"Raw response text:\n```\n{response.text}\n```")
        
        # Update run status with response
        if st.session_state.last_run_id:
            run_info = st.session_state.model_runs[st.session_state.last_run_id]
            run_info['response_time'] = datetime.now().isoformat()
        
        response.raise_for_status()
        
        # Special handling for ProteinMPNN responses
        if "proteinmpnn" in endpoint.lower():
            result = parse_proteinmpnn_response(response.text)
        else:
            # Try to parse JSON response for other models
            try:
                result = response.json()
            except json.JSONDecodeError:
                result = {"response": response.text}
        
        # Update run status
        if st.session_state.last_run_id:
            run_info = st.session_state.model_runs[st.session_state.last_run_id]
            run_info['status'] = 'completed'
            run_info['status_code'] = response.status_code
            
            if isinstance(result, dict):
                if "error" in result:
                    run_info['status'] = 'failed'
                    run_info['error'] = result["error"]
                else:
                    run_info['result_summary'] = {
                        'success': True,
                        'sequences': result.get("", []) if "" in result else []
                    }
        
        return result
        
    except requests.Timeout:
        error_msg = "Request timed out after 5 minutes"
        st.error(f"API Error: {error_msg}")
        
        if st.session_state.last_run_id:
            run_info = st.session_state.model_runs[st.session_state.last_run_id]
            run_info['status'] = 'failed'
            run_info['error'] = error_msg
        
        return {"error": error_msg}
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"API Error: {error_msg}")
        
        if st.session_state.last_run_id:
            run_info = st.session_state.model_runs[st.session_state.last_run_id]
            run_info['status'] = 'failed'
            run_info['error'] = error_msg
        
        return {"error": error_msg}

def main():
    st.title("🧬 BioML Models Dashboard")
    st.markdown("""
    This dashboard allows you to interact with various bio ML models deployed on the cloud.
    Select a model from the dropdown below and provide the required inputs.
    """)
    
    # AWS Credentials Section
    aws_credentials = create_aws_credentials_section()
    
    # Model Selection
    selected_model = st.selectbox(
        "Select Model",
        options=list(MODEL_ENDPOINTS.keys()),
        key="model_selector"
    )
    
    # Model-specific forms and logic
    st.subheader(f"Model: {selected_model}")
    
    params = None
    if selected_model == "Boltz":
        st.info("Upload a FASTA file containing the protein sequence.")
        params = create_boltz_form(aws_credentials)
    
    elif selected_model == "ProGPT2":
        params = create_progpt2_form(aws_credentials)
    
    elif selected_model == "ProteinMPNN":
        st.info("Upload one or more PDB files and set the generation parameters.")
        params = create_proteinmpnn_form(aws_credentials)
    
    elif selected_model == "RFDiffusion":
        st.info("Specify a contig and optionally upload a reference PDB file.")
        params = create_rfdiffusion_form(aws_credentials)
    
    elif selected_model == "ColabFold":
        params = create_colabfold_form(aws_credentials)
    
    # Handle API call and results
    if params:
        with st.spinner(f"Running {selected_model}..."):
            result = call_model_api(MODEL_ENDPOINTS[selected_model], params)
            
            # Display results section
            st.subheader("Results")
            
            # Show execution timeline if available
            if st.session_state.last_run_id:
                run_info = st.session_state.model_runs[st.session_state.last_run_id]
                duration = format_duration(run_info['start_time'])
                
                st.write("📊 Execution Timeline:")
                st.write(f"- Started: {run_info['start_time']}")
                st.write(f"- Total Duration: {duration}")
                if 'request_time' in run_info:
                    st.write(f"- API Request Sent: {run_info['request_time']}")
                if 'response_time' in run_info:
                    st.write(f"- Response Received: {run_info['response_time']}")
                st.write(f"- Status: {run_info['status'].upper()}")
                
                if 'error' in run_info:
                    st.error(f"Error: {run_info['error']}")
            
            # Display the results based on type
            if isinstance(result, dict):
                if "error" in result:
                    st.error(result["error"])
                elif "" in result and isinstance(result[""], list):
                    # Special handling for ProteinMPNN output
                    sequences = result[""]
                    st.markdown("**🧬 Generated Sequences:**")
                    
                    # Group sequences and metadata
                    current_header = None
                    for i in range(0, len(sequences), 2):
                        header = sequences[i]
                        sequence = sequences[i + 1] if i + 1 < len(sequences) else ""
                        
                        # Display in an expander
                        with st.expander(header.strip(">")):
                            # Show the sequence
                            st.text(sequence)
                            
                            # Parse and display metadata
                            metadata = {}
                            for part in header.strip(">").split(", "):
                                if "=" in part:
                                    key, value = part.split("=", 1)
                                    try:
                                        # Try to parse as Python literal for lists, etc.
                                        if value.startswith("[") or value.startswith("{"):
                                            metadata[key] = eval(value)
                                        elif value.replace(".", "").isdigit():
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = value
                                    except:
                                        metadata[key] = value
                            
                            if metadata:
                                st.json(metadata)
                else:
                    st.json(result)
            else:
                st.json(result)

if __name__ == "__main__":
    main()
