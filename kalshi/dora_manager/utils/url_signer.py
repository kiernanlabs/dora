"""
URL Signer for Secure Proposal Approval Links

Generates and validates HMAC-signed URLs with expiry timestamps.
"""
import hmac
import hashlib
import time
import logging
import boto3
from typing import Optional, Tuple
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class URLSigner:
    """Generate and validate HMAC-signed URLs for proposal approval."""

    def __init__(self, region: str = "us-east-1", environment: str = "prod"):
        """Initialize with secret key from Secrets Manager.

        Args:
            region: AWS region
            environment: 'demo' or 'prod'
        """
        self.region = region
        self.environment = environment
        self.secret_key = self._get_secret_key()

    def _get_secret_key(self) -> str:
        """Retrieve HMAC secret key from AWS Secrets Manager.

        Returns:
            Secret key string
        """
        secret_name = f"dora-manager/url-signing-key/{self.environment}"

        try:
            secrets_client = boto3.client('secretsmanager', region_name=self.region)
            response = secrets_client.get_secret_value(SecretId=secret_name)
            secret_key = response['SecretString']
            logger.info(f"Retrieved signing key from Secrets Manager: {secret_name}")
            return secret_key

        except ClientError as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise

    def generate_signature(
        self,
        proposal_id: str,
        expiry_timestamp: int
    ) -> str:
        """Generate HMAC signature for a proposal ID and expiry.

        Args:
            proposal_id: UUID of the proposal batch
            expiry_timestamp: Unix timestamp when signature expires

        Returns:
            Hex-encoded HMAC signature
        """
        # Create payload: proposal_id:expiry
        payload = f"{proposal_id}:{expiry_timestamp}"

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def generate_signed_url(
        self,
        base_url: str,
        proposal_id: str,
        endpoint: str = "",
        ttl_hours: int = 12
    ) -> str:
        """Generate a signed URL for proposal approval.

        Args:
            base_url: API Gateway base URL (e.g., https://xxx.execute-api.us-east-1.amazonaws.com/prod)
            proposal_id: UUID of the proposal batch
            endpoint: Optional endpoint suffix (e.g., '/execute' for direct approval)
            ttl_hours: Hours until URL expires (default: 12)

        Returns:
            Signed URL with signature and expiry parameters
        """
        # Calculate expiry timestamp
        expiry = int(time.time()) + (ttl_hours * 3600)

        # Generate signature
        signature = self.generate_signature(proposal_id, expiry)

        # Build URL
        url = f"{base_url}/proposals/{proposal_id}{endpoint}?signature={signature}&expiry={expiry}"

        logger.info(f"Generated signed URL for proposal {proposal_id}, expires at {expiry}")
        return url

    def validate_signature(
        self,
        proposal_id: str,
        signature: str,
        expiry: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate a URL signature and check expiry.

        Args:
            proposal_id: UUID of the proposal batch
            signature: HMAC signature from URL
            expiry: Expiry timestamp from URL

        Returns:
            Tuple of (is_valid, error_message)
            - (True, None) if valid
            - (False, error_message) if invalid
        """
        try:
            # Parse expiry timestamp
            expiry_timestamp = int(expiry)

            # Check if expired
            current_time = int(time.time())
            if current_time > expiry_timestamp:
                error_msg = f"URL expired at timestamp {expiry_timestamp}"
                logger.warning(error_msg)
                return False, error_msg

            # Generate expected signature
            expected_signature = self.generate_signature(proposal_id, expiry_timestamp)

            # Constant-time comparison to prevent timing attacks
            is_valid = hmac.compare_digest(signature, expected_signature)

            if not is_valid:
                error_msg = "Invalid signature"
                logger.warning(f"Signature validation failed for proposal {proposal_id}")
                return False, error_msg

            logger.info(f"Signature validated for proposal {proposal_id}")
            return True, None

        except ValueError as e:
            error_msg = f"Invalid expiry format: {e}"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            error_msg = f"Signature validation error: {e}"
            logger.error(error_msg)
            return False, error_msg
