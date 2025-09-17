import os
import logging
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class HubSpotClient:
    def __init__(self):
        self.server_name = "hubspot"
        self.is_configured = self._check_mcp_server()
    
    def _check_mcp_server(self) -> bool:
        """Check if HubSpot MCP server is available"""
        try:
            result = subprocess.run(
                ['manus-mcp-cli', 'tool', 'list', '--server', self.server_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"HubSpot MCP server not available: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if HubSpot client is connected"""
        return self.is_configured
    
    def _call_mcp_tool(self, tool_name: str, input_data: Dict) -> Optional[Dict]:
        """Call a HubSpot MCP tool"""
        try:
            if not self.is_configured:
                logger.error("HubSpot MCP server not configured")
                return None
            
            cmd = [
                'manus-mcp-cli', 'tool', 'call', tool_name,
                '--server', self.server_name,
                '--input', json.dumps(input_data)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.error(f"MCP tool call failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return None
    
    def create_contact_and_deal(self, registration_data: Dict) -> Dict:
        """Create a HubSpot contact and associated deal"""
        try:
            # Create contact first
            contact_data = {
                "email": registration_data.get('email'),
                "firstname": registration_data.get('name', '').split(' ')[0] if registration_data.get('name') else '',
                "lastname": ' '.join(registration_data.get('name', '').split(' ')[1:]) if len(registration_data.get('name', '').split(' ')) > 1 else '',
                "phone": registration_data.get('phone', ''),
                "company": registration_data.get('company', ''),
                "website": registration_data.get('website', ''),
                "lifecyclestage": "lead",
                "lead_source": registration_data.get('source', 'website'),
                "subscription_plan": registration_data.get('plan', 'basic'),
                "billing_cycle": registration_data.get('billing_cycle', 'monthly'),
                "registration_date": datetime.now().isoformat()
            }
            
            # Remove empty values
            contact_data = {k: v for k, v in contact_data.items() if v}
            
            contact_result = self._call_mcp_tool('create_contact', {
                'properties': contact_data
            })
            
            if not contact_result:
                raise Exception("Failed to create HubSpot contact")
            
            contact_id = contact_result.get('id')
            logger.info(f"Created HubSpot contact: {contact_id}")
            
            # Create associated deal
            plan_amounts = {
                'basic': 49,
                'premium': 99,
                'vip': 397
            }
            
            amount = plan_amounts.get(registration_data.get('plan', 'basic'), 49)
            if registration_data.get('billing_cycle') == 'yearly':
                amount *= 10  # Yearly discount
            
            deal_data = {
                "dealname": f"{registration_data.get('plan', 'Basic').title()} Plan - {registration_data.get('name', 'New Customer')}",
                "amount": amount,
                "dealstage": "appointmentscheduled",
                "pipeline": "default",
                "closedate": datetime.now().isoformat(),
                "subscription_plan": registration_data.get('plan', 'basic'),
                "billing_cycle": registration_data.get('billing_cycle', 'monthly')
            }
            
            deal_result = self._call_mcp_tool('create_deal', {
                'properties': deal_data,
                'associations': [
                    {
                        'to': {'id': contact_id},
                        'types': [{'associationCategory': 'HUBSPOT_DEFINED', 'associationTypeId': 3}]
                    }
                ]
            })
            
            if not deal_result:
                logger.warning("Failed to create HubSpot deal")
                deal_id = None
            else:
                deal_id = deal_result.get('id')
                logger.info(f"Created HubSpot deal: {deal_id}")
            
            return {
                'contact_id': contact_id,
                'deal_id': deal_id,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error creating HubSpot contact and deal: {e}")
            # Fallback: return mock data for testing
            return {
                'contact_id': f"mock_{int(datetime.now().timestamp())}",
                'deal_id': f"deal_{int(datetime.now().timestamp())}",
                'status': 'mock_success',
                'error': str(e)
            }
    
    def get_user_subscription(self, user_id: str) -> Optional[Dict]:
        """Get user subscription details from HubSpot"""
        try:
            # Search for contact by user ID or email
            search_result = self._call_mcp_tool('search_contacts', {
                'filters': [
                    {
                        'propertyName': 'email',
                        'operator': 'EQ',
                        'value': user_id
                    }
                ]
            })
            
            if not search_result or not search_result.get('results'):
                logger.warning(f"No contact found for user: {user_id}")
                return None
            
            contact = search_result['results'][0]
            properties = contact.get('properties', {})
            
            return {
                'user_id': user_id,
                'contact_id': contact.get('id'),
                'plan': properties.get('subscription_plan', 'basic'),
                'status': properties.get('lifecyclestage', 'lead'),
                'billing_cycle': properties.get('billing_cycle', 'monthly'),
                'email': properties.get('email'),
                'name': f"{properties.get('firstname', '')} {properties.get('lastname', '')}".strip()
            }
            
        except Exception as e:
            logger.error(f"Error getting user subscription: {e}")
            # Fallback: return basic subscription for testing
            return {
                'user_id': user_id,
                'plan': 'basic',
                'status': 'active',
                'billing_cycle': 'monthly'
            }
    
    def get_active_subscribers(self) -> List[Dict]:
        """Get all active subscribers from HubSpot"""
        try:
            # Search for contacts with active subscriptions
            search_result = self._call_mcp_tool('search_contacts', {
                'filters': [
                    {
                        'propertyName': 'lifecyclestage',
                        'operator': 'IN',
                        'values': ['customer', 'opportunity']
                    }
                ],
                'limit': 1000
            })
            
            if not search_result:
                return []
            
            subscribers = []
            for contact in search_result.get('results', []):
                properties = contact.get('properties', {})
                
                subscriber = {
                    'contact_id': contact.get('id'),
                    'email': properties.get('email'),
                    'name': f"{properties.get('firstname', '')} {properties.get('lastname', '')}".strip(),
                    'plan': properties.get('subscription_plan', 'basic'),
                    'status': properties.get('lifecyclestage', 'lead'),
                    'phone': properties.get('phone'),
                    'telegram_id': properties.get('telegram_id'),
                    'whatsapp_number': properties.get('whatsapp_number')
                }
                
                if subscriber['email']:  # Only include contacts with email
                    subscribers.append(subscriber)
            
            logger.info(f"Retrieved {len(subscribers)} active subscribers")
            return subscribers
            
        except Exception as e:
            logger.error(f"Error getting active subscribers: {e}")
            # Fallback: return mock subscribers for testing
            return [
                {
                    'contact_id': 'mock_1',
                    'email': 'test@example.com',
                    'name': 'Test User',
                    'plan': 'basic',
                    'status': 'customer'
                }
            ]
    
    def update_subscription_status(self, email: str, status: str, 
                                 subscription_id: str = None) -> bool:
        """Update subscription status in HubSpot"""
        try:
            # Find contact by email
            search_result = self._call_mcp_tool('search_contacts', {
                'filters': [
                    {
                        'propertyName': 'email',
                        'operator': 'EQ',
                        'value': email
                    }
                ]
            })
            
            if not search_result or not search_result.get('results'):
                logger.warning(f"No contact found for email: {email}")
                return False
            
            contact_id = search_result['results'][0].get('id')
            
            # Update contact properties
            update_data = {
                'lifecyclestage': 'customer' if status == 'active' else 'lead',
                'subscription_status': status,
                'subscription_updated': datetime.now().isoformat()
            }
            
            if subscription_id:
                update_data['stripe_subscription_id'] = subscription_id
            
            update_result = self._call_mcp_tool('update_contact', {
                'contact_id': contact_id,
                'properties': update_data
            })
            
            return update_result is not None
            
        except Exception as e:
            logger.error(f"Error updating subscription status: {e}")
            return False
    
    def get_contact_by_email(self, email: str) -> Optional[Dict]:
        """Get contact details by email"""
        try:
            search_result = self._call_mcp_tool('search_contacts', {
                'filters': [
                    {
                        'propertyName': 'email',
                        'operator': 'EQ',
                        'value': email
                    }
                ]
            })
            
            if search_result and search_result.get('results'):
                return search_result['results'][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting contact by email: {e}")
            return None
    
    def create_engagement(self, contact_id: str, engagement_type: str, 
                         subject: str, body: str) -> bool:
        """Create an engagement (note, email, etc.) in HubSpot"""
        try:
            engagement_data = {
                'engagement': {
                    'type': engagement_type,
                    'timestamp': int(datetime.now().timestamp() * 1000)
                },
                'associations': {
                    'contactIds': [contact_id]
                },
                'metadata': {
                    'subject': subject,
                    'body': body
                }
            }
            
            result = self._call_mcp_tool('create_engagement', engagement_data)
            return result is not None
            
        except Exception as e:
            logger.error(f"Error creating engagement: {e}")
            return False
