import os
import logging
import smtplib
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        
        # Telegram configuration
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_channel_id = os.getenv('TELEGRAM_CHANNEL_ID', '')
        
        # WhatsApp configuration (using WhatsApp Business API)
        self.whatsapp_token = os.getenv('WHATSAPP_TOKEN', '')
        self.whatsapp_phone_id = os.getenv('WHATSAPP_PHONE_ID', '')
        
        # SendGrid configuration (alternative to SMTP)
        self.sendgrid_api_key = os.getenv('SENDGRID_API_KEY', '')
    
    def is_ready(self) -> bool:
        """Check if notification service is configured"""
        return bool(self.email_user or self.sendgrid_api_key)
    
    def send_welcome_message(self, email: str, name: str, plan: str) -> bool:
        """Send welcome message to new subscriber"""
        try:
            subject = f"Welcome to QuantCompass - {plan.title()} Plan Activated!"
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #2563eb;">Welcome to QuantCompass!</h1>
                    </div>
                    
                    <p>Dear {name},</p>
                    
                    <p>Thank you for subscribing to our <strong>{plan.title()} Plan</strong>! You now have access to our premium trading signals.</p>
                    
                    <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <h3 style="color: #1e40af; margin-top: 0;">What's Next?</h3>
                        <ul>
                            <li>You'll receive real-time trading signals via email</li>
                            <li>Access your dashboard for detailed performance metrics</li>
                            <li>Join our Telegram channel for instant notifications</li>
                            <li>Get WhatsApp alerts for high-priority signals</li>
                        </ul>
                    </div>
                    
                    <div style="background-color: #ecfdf5; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <h3 style="color: #059669; margin-top: 0;">Your Plan Benefits:</h3>
                        {self._get_plan_benefits_html(plan)}
                    </div>
                    
                    <p>If you have any questions, please don't hesitate to contact our support team.</p>
                    
                    <p>Happy Trading!<br>
                    The QuantCompass Team</p>
                    
                    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                        <p style="color: #6b7280; font-size: 12px;">
                            This email was sent to {email}. You're receiving this because you subscribed to QuantCompass.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(email, subject, html_content)
            
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}")
            return False
    
    def send_activation_message(self, email: str) -> bool:
        """Send subscription activation message"""
        try:
            subject = "QuantCompass Subscription Activated - Start Receiving Signals!"
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #059669;">🎉 Subscription Activated!</h1>
                    </div>
                    
                    <p>Great news! Your QuantCompass subscription has been successfully activated.</p>
                    
                    <div style="background-color: #ecfdf5; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center;">
                        <h3 style="color: #059669; margin-top: 0;">You're all set to receive trading signals!</h3>
                        <p>Our AI-powered system is now monitoring the markets for you.</p>
                    </div>
                    
                    <p>You'll start receiving trading signals within the next few hours. Each signal includes:</p>
                    <ul>
                        <li>Entry price and timing</li>
                        <li>Stop loss and take profit levels</li>
                        <li>Confidence score and analysis</li>
                        <li>Risk management guidelines</li>
                    </ul>
                    
                    <p>Best regards,<br>
                    The QuantCompass Team</p>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(email, subject, html_content)
            
        except Exception as e:
            logger.error(f"Error sending activation message: {e}")
            return False
    
    def send_test_signal(self, email: str, signal: Dict) -> bool:
        """Send a test signal to specific email"""
        try:
            subject = f"🧪 Test Signal: {signal['type']} {signal['symbol']}"
            
            html_content = self._format_signal_email([signal], is_test=True)
            
            return self._send_email(email, subject, html_content)
            
        except Exception as e:
            logger.error(f"Error sending test signal: {e}")
            return False
    
    def broadcast_signals(self, signals: List[Dict], subscribers: List[Dict]) -> Dict:
        """Broadcast signals to all subscribers via multiple channels"""
        try:
            results = {
                'email': {'sent': 0, 'failed': 0},
                'telegram': {'sent': 0, 'failed': 0},
                'whatsapp': {'sent': 0, 'failed': 0}
            }
            
            # Group subscribers by plan
            plan_groups = {}
            for subscriber in subscribers:
                plan = subscriber.get('plan', 'basic')
                if plan not in plan_groups:
                    plan_groups[plan] = []
                plan_groups[plan].append(subscriber)
            
            # Send signals to each plan group
            for plan, plan_subscribers in plan_groups.items():
                # Filter signals for this plan
                plan_signals = self._filter_signals_for_plan(signals, plan)
                
                if not plan_signals:
                    continue
                
                # Send via email
                for subscriber in plan_subscribers:
                    if subscriber.get('email'):
                        if self._send_signal_email(subscriber['email'], plan_signals):
                            results['email']['sent'] += 1
                        else:
                            results['email']['failed'] += 1
                
                # Send via Telegram
                if self.telegram_bot_token and self.telegram_channel_id:
                    if self._send_telegram_signals(plan_signals):
                        results['telegram']['sent'] += len(plan_subscribers)
                    else:
                        results['telegram']['failed'] += len(plan_subscribers)
                
                # Send via WhatsApp (for VIP subscribers)
                if plan == 'vip':
                    for subscriber in plan_subscribers:
                        whatsapp_number = subscriber.get('whatsapp_number')
                        if whatsapp_number:
                            if self._send_whatsapp_signals(whatsapp_number, plan_signals):
                                results['whatsapp']['sent'] += 1
                            else:
                                results['whatsapp']['failed'] += 1
            
            logger.info(f"Signal broadcast completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error broadcasting signals: {e}")
            return {'error': str(e)}
    
    def send_real_time_signals(self, signals: List[Dict], subscribers: List[Dict]) -> bool:
        """Send real-time signals (high priority)"""
        try:
            # For real-time signals, prioritize speed
            # Send via Telegram first (fastest)
            if self.telegram_bot_token:
                self._send_telegram_signals(signals, urgent=True)
            
            # Send via email to VIP subscribers
            vip_subscribers = [s for s in subscribers if s.get('plan') == 'vip']
            for subscriber in vip_subscribers:
                if subscriber.get('email'):
                    self._send_signal_email(subscriber['email'], signals, urgent=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending real-time signals: {e}")
            return False
    
    def _send_email(self, to_email: str, subject: str, html_content: str) -> bool:
        """Send email using SMTP or SendGrid"""
        try:
            if self.sendgrid_api_key:
                return self._send_via_sendgrid(to_email, subject, html_content)
            else:
                return self._send_via_smtp(to_email, subject, html_content)
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _send_via_smtp(self, to_email: str, subject: str, html_content: str) -> bool:
        """Send email via SMTP"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_user
            msg['To'] = to_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            return False
    
    def _send_via_sendgrid(self, to_email: str, subject: str, html_content: str) -> bool:
        """Send email via SendGrid API"""
        try:
            url = "https://api.sendgrid.com/v3/mail/send"
            
            data = {
                "personalizations": [
                    {
                        "to": [{"email": to_email}],
                        "subject": subject
                    }
                ],
                "from": {"email": self.email_user, "name": "QuantCompass"},
                "content": [
                    {
                        "type": "text/html",
                        "value": html_content
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.sendgrid_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=data, headers=headers)
            return response.status_code == 202
            
        except Exception as e:
            logger.error(f"SendGrid error: {e}")
            return False
    
    def _send_signal_email(self, email: str, signals: List[Dict], urgent: bool = False) -> bool:
        """Send trading signals via email"""
        try:
            if urgent:
                subject = f"🚨 URGENT: {len(signals)} New Trading Signal{'s' if len(signals) > 1 else ''}"
            else:
                subject = f"📈 {len(signals)} New Trading Signal{'s' if len(signals) > 1 else ''} - QuantCompass"
            
            html_content = self._format_signal_email(signals, urgent=urgent)
            
            return self._send_email(email, subject, html_content)
            
        except Exception as e:
            logger.error(f"Error sending signal email: {e}")
            return False
    
    def _send_telegram_signals(self, signals: List[Dict], urgent: bool = False) -> bool:
        """Send signals via Telegram"""
        try:
            if not self.telegram_bot_token or not self.telegram_channel_id:
                return False
            
            message = self._format_telegram_message(signals, urgent)
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": self.telegram_channel_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=data)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def _send_whatsapp_signals(self, phone_number: str, signals: List[Dict]) -> bool:
        """Send signals via WhatsApp Business API"""
        try:
            if not self.whatsapp_token or not self.whatsapp_phone_id:
                return False
            
            message = self._format_whatsapp_message(signals)
            
            url = f"https://graph.facebook.com/v17.0/{self.whatsapp_phone_id}/messages"
            headers = {
                "Authorization": f"Bearer {self.whatsapp_token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messaging_product": "whatsapp",
                "to": phone_number,
                "type": "text",
                "text": {"body": message}
            }
            
            response = requests.post(url, json=data, headers=headers)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"WhatsApp error: {e}")
            return False
    
    def _format_signal_email(self, signals: List[Dict], urgent: bool = False, is_test: bool = False) -> str:
        """Format signals for email"""
        signals_html = ""
        for signal in signals:
            signal_color = "#059669" if signal['type'] == 'BUY' else "#dc2626"
            
            signals_html += f"""
            <div style="border: 2px solid {signal_color}; border-radius: 8px; padding: 20px; margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3 style="color: {signal_color}; margin: 0;">{signal['type']} {signal['symbol']}</h3>
                    <span style="background-color: {signal_color}; color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px;">
                        {signal['confidence']}% Confidence
                    </span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <strong>Entry Price:</strong> {signal['entry_price']}<br>
                        <strong>Stop Loss:</strong> {signal['stop_loss']}<br>
                        <strong>Take Profit:</strong> {signal['take_profit']}
                    </div>
                    <div>
                        <strong>Timeframe:</strong> {signal['timeframe']}<br>
                        <strong>Generated:</strong> {datetime.fromisoformat(signal['timestamp']).strftime('%H:%M:%S')}<br>
                        <strong>Model:</strong> v{signal['model_version']}
                    </div>
                </div>
            </div>
            """
        
        test_banner = """
        <div style="background-color: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center;">
            <h3 style="color: #92400e; margin: 0;">🧪 TEST SIGNAL</h3>
            <p style="color: #92400e; margin: 5px 0 0 0;">This is a test signal for demonstration purposes.</p>
        </div>
        """ if is_test else ""
        
        urgent_banner = """
        <div style="background-color: #fef2f2; border: 2px solid #dc2626; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center;">
            <h3 style="color: #dc2626; margin: 0;">🚨 URGENT SIGNAL</h3>
            <p style="color: #dc2626; margin: 5px 0 0 0;">Time-sensitive trading opportunity - act quickly!</p>
        </div>
        """ if urgent else ""
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #2563eb;">QuantCompass Trading Signals</h1>
                </div>
                
                {test_banner}
                {urgent_banner}
                
                {signals_html}
                
                <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h4 style="color: #1e40af; margin-top: 0;">⚠️ Risk Disclaimer</h4>
                    <p style="font-size: 12px; color: #6b7280;">
                        Trading involves substantial risk and may not be suitable for all investors. 
                        Past performance is not indicative of future results. Please trade responsibly.
                    </p>
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <p style="color: #6b7280; font-size: 12px;">
                        Generated by QuantCompass AI • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _format_telegram_message(self, signals: List[Dict], urgent: bool = False) -> str:
        """Format signals for Telegram"""
        header = "🚨 <b>URGENT SIGNALS</b> 🚨\n\n" if urgent else "📈 <b>New Trading Signals</b>\n\n"
        
        message = header
        for signal in signals:
            emoji = "🟢" if signal['type'] == 'BUY' else "🔴"
            message += f"{emoji} <b>{signal['type']} {signal['symbol']}</b>\n"
            message += f"💰 Entry: {signal['entry_price']}\n"
            message += f"🛑 SL: {signal['stop_loss']}\n"
            message += f"🎯 TP: {signal['take_profit']}\n"
            message += f"📊 Confidence: {signal['confidence']}%\n"
            message += f"⏰ {datetime.fromisoformat(signal['timestamp']).strftime('%H:%M')}\n\n"
        
        message += "⚠️ <i>Trade responsibly. Risk management is key.</i>"
        return message
    
    def _format_whatsapp_message(self, signals: List[Dict]) -> str:
        """Format signals for WhatsApp"""
        message = "📈 *QuantCompass VIP Signals*\n\n"
        
        for signal in signals:
            emoji = "🟢" if signal['type'] == 'BUY' else "🔴"
            message += f"{emoji} *{signal['type']} {signal['symbol']}*\n"
            message += f"Entry: {signal['entry_price']}\n"
            message += f"SL: {signal['stop_loss']}\n"
            message += f"TP: {signal['take_profit']}\n"
            message += f"Confidence: {signal['confidence']}%\n\n"
        
        message += "_Trade responsibly. This is not financial advice._"
        return message
    
    def _filter_signals_for_plan(self, signals: List[Dict], plan: str) -> List[Dict]:
        """Filter signals based on subscription plan"""
        plan_symbols = {
            'basic': ['EURUSD', 'GBPUSD', 'USDJPY'],
            'premium': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD'],
            'vip': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP']
        }
        
        allowed_symbols = plan_symbols.get(plan, plan_symbols['basic'])
        return [signal for signal in signals if signal['symbol'] in allowed_symbols]
    
    def _get_plan_benefits_html(self, plan: str) -> str:
        """Get HTML for plan benefits"""
        benefits = {
            'basic': [
                "Up to 5 signals per day",
                "3 major currency pairs",
                "Email delivery",
                "Basic performance metrics"
            ],
            'premium': [
                "Up to 15 signals per day",
                "5 currency pairs",
                "Email + Telegram delivery",
                "Advanced analytics",
                "Priority support"
            ],
            'vip': [
                "Up to 50 signals per day",
                "10 currency pairs",
                "Email + Telegram + WhatsApp",
                "Real-time notifications",
                "Detailed market analysis",
                "1-on-1 support"
            ]
        }
        
        plan_benefits = benefits.get(plan, benefits['basic'])
        return "<ul>" + "".join([f"<li>{benefit}</li>" for benefit in plan_benefits]) + "</ul>"
