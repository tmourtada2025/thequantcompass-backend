from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime, timedelta
import json
import asyncio
import threading
import time
from typing import Dict, List, Optional

# Import our modules
from polygon_client import PolygonClient
from signal_generator import SignalGenerator
from hubspot_client import HubSpotClient
from notification_service import NotificationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize services
polygon_client = PolygonClient()
signal_generator = SignalGenerator()
hubspot_client = HubSpotClient()
notification_service = NotificationService()

# Global state for real-time data
latest_signals = []
active_subscriptions = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'quantcompass-gcp-backend',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'components': {
            'polygon': polygon_client.is_connected(),
            'signal_generator': signal_generator.is_ready(),
            'hubspot': hubspot_client.is_connected(),
            'notifications': notification_service.is_ready()
        }
    }), 200

@app.route('/api/signals/latest', methods=['GET'])
def get_latest_signals():
    """Get the latest trading signals"""
    try:
        # Get user authentication from request
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        # Check user subscription status via HubSpot
        subscription = hubspot_client.get_user_subscription(user_id)
        if not subscription or subscription.get('status') != 'active':
            return jsonify({'error': 'Active subscription required'}), 403
        
        # Filter signals based on subscription plan
        plan = subscription.get('plan', 'basic')
        filtered_signals = signal_generator.filter_signals_by_plan(latest_signals, plan)
        
        return jsonify({
            'signals': filtered_signals,
            'count': len(filtered_signals),
            'plan': plan,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting latest signals: {e}")
        return jsonify({'error': 'Failed to retrieve signals'}), 500

@app.route('/api/signals/history', methods=['GET'])
def get_signal_history():
    """Get historical signals with pagination"""
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        symbol = request.args.get('symbol')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Check subscription
        subscription = hubspot_client.get_user_subscription(user_id)
        if not subscription or subscription.get('status') != 'active':
            return jsonify({'error': 'Active subscription required'}), 403
        
        # Get historical signals
        signals = signal_generator.get_historical_signals(
            page=page,
            limit=limit,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            plan=subscription.get('plan', 'basic')
        )
        
        return jsonify(signals), 200
        
    except Exception as e:
        logger.error(f"Error getting signal history: {e}")
        return jsonify({'error': 'Failed to retrieve signal history'}), 500

@app.route('/api/signals/performance', methods=['GET'])
def get_signal_performance():
    """Get signal performance metrics"""
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        subscription = hubspot_client.get_user_subscription(user_id)
        if not subscription or subscription.get('status') != 'active':
            return jsonify({'error': 'Active subscription required'}), 403
        
        # Get performance metrics
        performance = signal_generator.get_performance_metrics(
            plan=subscription.get('plan', 'basic')
        )
        
        return jsonify(performance), 200
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({'error': 'Failed to retrieve performance metrics'}), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    """Register a new user via HubSpot"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('email') or not data.get('name'):
            return jsonify({'error': 'Email and name are required'}), 400
        
        logger.info(f"Processing registration for: {data.get('email')}")
        
        # Create HubSpot contact and deal
        result = hubspot_client.create_contact_and_deal(data)
        
        # Send welcome notification
        notification_service.send_welcome_message(
            email=data.get('email'),
            name=data.get('name'),
            plan=data.get('plan', 'basic')
        )
        
        return jsonify({
            'message': 'Registration successful',
            'contact_id': result.get('contact_id'),
            'deal_id': result.get('deal_id'),
            'status': 'registered'
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed - please try again'}), 500

@app.route('/api/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks for subscription updates"""
    try:
        # TODO: Verify Stripe webhook signature
        data = request.get_json()
        
        if data.get('type') == 'checkout.session.completed':
            session = data.get('data', {}).get('object', {})
            customer_email = session.get('customer_details', {}).get('email')
            
            if customer_email:
                # Update HubSpot contact with payment success
                hubspot_client.update_subscription_status(
                    email=customer_email,
                    status='active',
                    subscription_id=session.get('subscription')
                )
                
                # Send activation notification
                notification_service.send_activation_message(customer_email)
                
                logger.info(f"Subscription activated for: {customer_email}")
        
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': 'Webhook processing failed'}), 500

@app.route('/api/admin/send-signals', methods=['POST'])
def send_signals_to_subscribers():
    """Admin endpoint to send signals to all active subscribers"""
    try:
        # TODO: Add admin authentication
        
        # Get all active subscribers from HubSpot
        subscribers = hubspot_client.get_active_subscribers()
        
        # Get latest signals
        signals = signal_generator.get_latest_signals()
        
        if not signals:
            return jsonify({'message': 'No new signals to send'}), 200
        
        # Send signals via multiple channels
        results = notification_service.broadcast_signals(signals, subscribers)
        
        return jsonify({
            'message': f'Signals sent to {len(subscribers)} subscribers',
            'signals_count': len(signals),
            'delivery_results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error sending signals: {e}")
        return jsonify({'error': 'Failed to send signals'}), 500

@app.route('/api/admin/test-signal', methods=['POST'])
def send_test_signal():
    """Admin endpoint to send test signals"""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email required'}), 400
        
        # Generate test signal
        test_signal = signal_generator.generate_test_signal()
        
        # Send test signal
        result = notification_service.send_test_signal(email, test_signal)
        
        return jsonify({
            'message': 'Test signal sent successfully',
            'signal': test_signal,
            'delivery_result': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error sending test signal: {e}")
        return jsonify({'error': 'Failed to send test signal'}), 500

def start_real_time_data_stream():
    """Start the real-time data stream from Polygon.io"""
    def run_stream():
        logger.info("Starting real-time data stream...")
        polygon_client.start_websocket_stream(on_data_received)
    
    # Run in separate thread
    stream_thread = threading.Thread(target=run_stream, daemon=True)
    stream_thread.start()

def on_data_received(market_data):
    """Callback for when new market data is received"""
    try:
        # Process market data through ML model
        signals = signal_generator.process_market_data(market_data)
        
        if signals:
            # Update global state
            global latest_signals
            latest_signals.extend(signals)
            
            # Keep only last 100 signals
            latest_signals = latest_signals[-100:]
            
            # Get active subscribers
            subscribers = hubspot_client.get_active_subscribers()
            
            # Send real-time signals
            notification_service.send_real_time_signals(signals, subscribers)
            
            logger.info(f"Generated and sent {len(signals)} new signals")
            
    except Exception as e:
        logger.error(f"Error processing market data: {e}")

if __name__ == '__main__':
    # Start real-time data stream
    start_real_time_data_stream()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
