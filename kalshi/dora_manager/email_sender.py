"""
SES Email Sender for Dora Manager Reports
"""
import boto3
from typing import Optional
import logging
from calculator import TradingSummary, MarketSummary

logger = logging.getLogger(__name__)


class EmailSender:
    """Send trading summary reports via AWS SES."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize SES client."""
        self.ses = boto3.client('ses', region_name=region)
        self.sender_email = "joe@kiernanlabs.com"  # Must be verified in SES
        self.default_recipient = "joe@kiernanlabs.com"

    def format_currency(self, value: Optional[float], include_sign: bool = False) -> str:
        """Format a value as currency."""
        if value is None:
            return "N/A"
        if include_sign:
            return f"${value:+,.2f}"
        return f"${value:,.2f}"

    def format_position_qty(self, value: Optional[object]) -> str:
        """Format a position quantity as a signed integer string."""
        if value is None:
            qty = 0
        else:
            try:
                qty = int(value)
            except (TypeError, ValueError):
                try:
                    qty = int(float(value))
                except (TypeError, ValueError):
                    qty = 0
        return f"{qty:+d}"

    def format_market_row_html(self, market: MarketSummary) -> str:
        """Format a market summary as an HTML table row."""
        pnl_window_color = "green" if market.realized_pnl_window >= 0 else "red"
        pnl_all_time_color = "green" if market.realized_pnl_all_time >= 0 else "red"

        flag_indicator = "&#9888; " if market.flagged_for_deactivation else ""

        unrealized_best_color = "green" if (market.unrealized_pnl_best or 0) >= 0 else "red"

        return f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{flag_indicator}{market.market_id}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {pnl_window_color}; font-weight: bold;">
                {self.format_currency(market.realized_pnl_window, include_sign=True)}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {pnl_all_time_color};">
                {self.format_currency(market.realized_pnl_all_time, include_sign=True)}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{market.trade_count}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{market.contracts_traded}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{market.net_position}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                {self.format_currency(market.unrealized_pnl_worst, include_sign=True)}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {unrealized_best_color}; font-weight: bold;">
                {self.format_currency(market.unrealized_pnl_best, include_sign=True)}
            </td>
        </tr>
        """

    def format_flagged_market_html(self, market: MarketSummary) -> str:
        """Format a flagged market as HTML."""
        return f"""
        <div style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 12px; margin-bottom: 8px;">
            <strong style="color: #856404;">&#9888; {market.market_id}</strong>
            <p style="margin: 8px 0 0 0; color: #856404;">
                {market.deactivation_reason}
            </p>
            <p style="margin: 4px 0 0 0; font-size: 12px; color: #666;">
                Trades: {market.trade_count} | Contracts: {market.contracts_traded} | Position: {market.net_position}
            </p>
        </div>
        """

    def build_html_report(self, summary: TradingSummary, environment: str) -> str:
        """Build HTML email body from trading summary."""
        # Determine overall status color
        if summary.markets_flagged_count > 0:
            status_color = "#dc3545"  # Red
            status_text = f"{summary.markets_flagged_count} Market(s) Flagged"
        elif summary.total_realized_pnl_window < 0:
            status_color = "#ffc107"  # Yellow
            status_text = "Negative P&L"
        else:
            status_color = "#28a745"  # Green
            status_text = "Healthy"

        pnl_window_color = "green" if summary.total_realized_pnl_window >= 0 else "red"
        pnl_all_time_color = "green" if summary.total_realized_pnl_all_time >= 0 else "red"

        # Build flagged markets section
        flagged_section = ""
        if summary.flagged_markets:
            flagged_html = "".join(
                self.format_flagged_market_html(m) for m in summary.flagged_markets
            )
            flagged_section = f"""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 16px; margin-bottom: 24px;">
                <h3 style="color: #721c24; margin-top: 0;">&#9888; Markets Flagged for Deactivation</h3>
                <p style="color: #721c24; margin-bottom: 12px;">
                    The following markets have been automatically disabled due to poor performance:
                </p>
                {flagged_html}
            </div>
            """

        # Build markets with trades table
        markets_table_rows = ""
        if summary.markets_with_window_trades:
            markets_table_rows = "".join(
                self.format_market_row_html(m) for m in summary.markets_with_window_trades
            )
        else:
            markets_table_rows = f"""
            <tr>
                <td colspan="8" style="padding: 16px; text-align: center; color: #666;">
                    No trades in the last {summary.window_hours} hours and no open positions
                </td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Dora Bot Trading Summary</title>
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;">
            <div style="background-color: white; border-radius: 8px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <!-- Header -->
                <div style="border-bottom: 2px solid #eee; padding-bottom: 16px; margin-bottom: 24px;">
                    <h1 style="margin: 0; color: #333;">Dora Bot Trading Summary</h1>
                    <p style="margin: 8px 0 0 0; color: #666;">
                        Environment: <strong>{environment.upper()}</strong> |
                        Last {summary.window_hours} Hours |
                        Generated: {summary.report_timestamp[:19].replace('T', ' ')} UTC
                    </p>
                    <div style="margin-top: 12px;">
                        <span style="background-color: {status_color}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">
                            {status_text}
                        </span>
                    </div>
                </div>

                {flagged_section}

                <!-- Summary Metrics -->
                <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px;">
                    <div style="flex: 1; min-width: 200px; background-color: #f8f9fa; border-radius: 8px; padding: 16px;">
                        <h4 style="margin: 0 0 8px 0; color: #666; font-size: 12px; text-transform: uppercase;">
                            P&L (Last {summary.window_hours}hrs)
                        </h4>
                        <p style="margin: 0; font-size: 24px; font-weight: bold; color: {pnl_window_color};">
                            {self.format_currency(summary.total_realized_pnl_window, include_sign=True)}
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background-color: #f8f9fa; border-radius: 8px; padding: 16px;">
                        <h4 style="margin: 0 0 8px 0; color: #666; font-size: 12px; text-transform: uppercase;">
                            P&L (All Time)
                        </h4>
                        <p style="margin: 0; font-size: 24px; font-weight: bold; color: {pnl_all_time_color};">
                            {self.format_currency(summary.total_realized_pnl_all_time, include_sign=True)}
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background-color: #f8f9fa; border-radius: 8px; padding: 16px;">
                        <h4 style="margin: 0 0 8px 0; color: #666; font-size: 12px; text-transform: uppercase;">
                            Total Exposure
                        </h4>
                        <p style="margin: 0; font-size: 24px; font-weight: bold; color: #333;">
                            {summary.total_exposure} contracts
                        </p>
                    </div>
                </div>

                <!-- Trade Stats -->
                <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px;">
                    <div style="flex: 1; min-width: 150px; background-color: #e3f2fd; border-radius: 8px; padding: 12px;">
                        <h4 style="margin: 0 0 4px 0; color: #1565c0; font-size: 11px; text-transform: uppercase;">
                            Trades ({summary.window_hours}hrs)
                        </h4>
                        <p style="margin: 0; font-size: 18px; font-weight: bold; color: #1565c0;">
                            {summary.total_trade_count}
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 150px; background-color: #e3f2fd; border-radius: 8px; padding: 12px;">
                        <h4 style="margin: 0 0 4px 0; color: #1565c0; font-size: 11px; text-transform: uppercase;">
                            Contracts Traded
                        </h4>
                        <p style="margin: 0; font-size: 18px; font-weight: bold; color: #1565c0;">
                            {summary.total_contracts_traded}
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 150px; background-color: #e3f2fd; border-radius: 8px; padding: 12px;">
                        <h4 style="margin: 0 0 4px 0; color: #1565c0; font-size: 11px; text-transform: uppercase;">
                            Fees Paid
                        </h4>
                        <p style="margin: 0; font-size: 18px; font-weight: bold; color: #1565c0;">
                            {self.format_currency(summary.total_fees_paid)}
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 150px; background-color: #e3f2fd; border-radius: 8px; padding: 12px;">
                        <h4 style="margin: 0 0 4px 0; color: #1565c0; font-size: 11px; text-transform: uppercase;">
                            Markets with Trades
                        </h4>
                        <p style="margin: 0; font-size: 18px; font-weight: bold; color: #1565c0;">
                            {summary.markets_with_trades} / {summary.active_markets_count}
                        </p>
                    </div>
                </div>

                <!-- Unrealized P&L -->
                <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px;">
                    <div style="flex: 1; min-width: 200px; background-color: #fff3e0; border-radius: 8px; padding: 12px;">
                        <h4 style="margin: 0 0 4px 0; color: #e65100; font-size: 11px; text-transform: uppercase;">
                            Unrealized P&L (Worst)
                        </h4>
                        <p style="margin: 0; font-size: 18px; font-weight: bold; color: #e65100;">
                            {self.format_currency(summary.total_unrealized_pnl_worst, include_sign=True)}
                        </p>
                        <p style="margin: 4px 0 0 0; font-size: 11px; color: #666;">
                            Exit at market best bid/ask
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background-color: #e8f5e9; border-radius: 8px; padding: 12px;">
                        <h4 style="margin: 0 0 4px 0; color: #2e7d32; font-size: 11px; text-transform: uppercase;">
                            Unrealized P&L (Best)
                        </h4>
                        <p style="margin: 0; font-size: 18px; font-weight: bold; color: #2e7d32;">
                            {self.format_currency(summary.total_unrealized_pnl_best, include_sign=True)}
                        </p>
                        <p style="margin: 4px 0 0 0; font-size: 11px; color: #666;">
                            Exit at our competitive orders
                        </p>
                    </div>
                </div>

                <!-- Active Orders -->
                <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px;">
                    <div style="flex: 1; min-width: 150px; background-color: #f3e5f5; border-radius: 8px; padding: 12px;">
                        <h4 style="margin: 0 0 4px 0; color: #7b1fa2; font-size: 11px; text-transform: uppercase;">
                            Active Bids
                        </h4>
                        <p style="margin: 0; font-size: 16px; font-weight: bold; color: #7b1fa2;">
                            {summary.active_bids_count} markets ({summary.active_bids_qty} cts)
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 150px; background-color: #f3e5f5; border-radius: 8px; padding: 12px;">
                        <h4 style="margin: 0 0 4px 0; color: #7b1fa2; font-size: 11px; text-transform: uppercase;">
                            Active Asks
                        </h4>
                        <p style="margin: 0; font-size: 16px; font-weight: bold; color: #7b1fa2;">
                            {summary.active_asks_count} markets ({summary.active_asks_qty} cts)
                        </p>
                    </div>
                </div>

                <!-- Markets with Trades Detail Table -->
                <h3 style="color: #333; margin-bottom: 12px;">Active Markets (Trades in Last {summary.window_hours} Hours or Open Positions)</h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                        <thead>
                            <tr style="background-color: #f8f9fa;">
                                <th style="padding: 10px 8px; text-align: left; border-bottom: 2px solid #dee2e6;">Market</th>
                                <th style="padding: 10px 8px; text-align: left; border-bottom: 2px solid #dee2e6;">P&L ({summary.window_hours}hrs)</th>
                                <th style="padding: 10px 8px; text-align: left; border-bottom: 2px solid #dee2e6;">P&L (All Time)</th>
                                <th style="padding: 10px 8px; text-align: left; border-bottom: 2px solid #dee2e6;">Trades</th>
                                <th style="padding: 10px 8px; text-align: left; border-bottom: 2px solid #dee2e6;">Contracts</th>
                                <th style="padding: 10px 8px; text-align: left; border-bottom: 2px solid #dee2e6;">Position</th>
                                <th style="padding: 10px 8px; text-align: left; border-bottom: 2px solid #dee2e6;">Unreal (W)</th>
                                <th style="padding: 10px 8px; text-align: left; border-bottom: 2px solid #dee2e6;">Unreal (B)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {markets_table_rows}
                        </tbody>
                    </table>
                </div>

                <!-- Footer -->
                <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #eee; text-align: center; color: #666; font-size: 12px;">
                    <p>This is an automated report from Dora Bot.</p>
                    <p>
                        <a href="https://dorabot.streamlit.app" style="color: #1a73e8;">View Full Dashboard</a>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def build_text_report(self, summary: TradingSummary, environment: str) -> str:
        """Build plain text email body from trading summary."""
        lines = [
            "=" * 60,
            "DORA BOT TRADING SUMMARY",
            "=" * 60,
            f"Environment: {environment.upper()}",
            f"Time Window: Last {summary.window_hours} hours",
            f"Generated: {summary.report_timestamp[:19].replace('T', ' ')} UTC",
            "",
        ]

        # Flagged markets warning
        if summary.flagged_markets:
            lines.append("!" * 60)
            lines.append("WARNING: MARKETS FLAGGED FOR DEACTIVATION")
            lines.append("!" * 60)
            for market in summary.flagged_markets:
                lines.append(f"  - {market.market_id}: {market.deactivation_reason}")
            lines.append("")

        # Summary metrics
        lines.extend([
            "-" * 60,
            "SUMMARY METRICS",
            "-" * 60,
            f"Realized P&L (Last {summary.window_hours}hrs): {self.format_currency(summary.total_realized_pnl_window, include_sign=True)}",
            f"Realized P&L (All Time):      {self.format_currency(summary.total_realized_pnl_all_time, include_sign=True)}",
            f"Unrealized P&L (Worst):       {self.format_currency(summary.total_unrealized_pnl_worst, include_sign=True)}",
            f"Unrealized P&L (Best):        {self.format_currency(summary.total_unrealized_pnl_best, include_sign=True)}",
            "",
            f"Total Exposure:    {summary.total_exposure} contracts",
            f"Trades:            {summary.total_trade_count}",
            f"Contracts Traded:  {summary.total_contracts_traded}",
            f"Fees Paid:         {self.format_currency(summary.total_fees_paid)}",
            "",
            f"Active Markets:    {summary.active_markets_count}",
            f"Markets with Trades: {summary.markets_with_trades}",
            f"Active Bids:       {summary.active_bids_count} markets ({summary.active_bids_qty} contracts)",
            f"Active Asks:       {summary.active_asks_count} markets ({summary.active_asks_qty} contracts)",
            "",
        ])

        # Markets with trades or open positions
        if summary.markets_with_window_trades:
            lines.extend([
                "-" * 60,
                f"ACTIVE MARKETS (Trades in Last {summary.window_hours} Hours or Open Positions)",
                "-" * 60,
            ])
            for market in summary.markets_with_window_trades:
                flag = "[FLAGGED] " if market.flagged_for_deactivation else ""
                lines.append(f"{flag}{market.market_id}")
                lines.append(f"  Event: {market.event_ticker or 'N/A'}")
                lines.append(f"  P&L ({summary.window_hours}hrs): {self.format_currency(market.realized_pnl_window, include_sign=True)}")
                lines.append(f"  P&L (All Time): {self.format_currency(market.realized_pnl_all_time, include_sign=True)}")
                lines.append(f"  Trades: {market.trade_count} | Contracts: {market.contracts_traded}")
                lines.append(f"  Position: {market.net_position} | Avg Cost: {self.format_currency(market.avg_cost)}")
                lines.append(f"  Unrealized (W/B): {self.format_currency(market.unrealized_pnl_worst, include_sign=True)} / {self.format_currency(market.unrealized_pnl_best, include_sign=True)}")
                lines.append("")
        else:
            lines.append(f"No trades in the last {summary.window_hours} hours and no open positions.")
            lines.append("")

        lines.extend([
            "=" * 60,
            "End of Report",
            "=" * 60,
        ])

        return "\n".join(lines)

    def send_report(
        self,
        summary: TradingSummary,
        environment: str,
        recipient: Optional[str] = None,
    ) -> bool:
        """
        Send trading summary report via SES.

        Args:
            summary: Trading summary data
            environment: Environment name (demo/prod)
            recipient: Email recipient (defaults to joey32@gmail.com)

        Returns:
            True if email sent successfully, False otherwise
        """
        recipient = recipient or self.default_recipient

        logger.info(f"Preparing to send email report")
        logger.info(f"  Sender: {self.sender_email}")
        logger.info(f"  Recipient: {recipient}")

        # Build subject line
        pnl_str = f"${summary.total_realized_pnl_window:+.2f}"
        trades_str = f"{summary.total_trade_count} trades"

        if summary.markets_flagged_count > 0:
            subject = f"[Dora Bot Summary Last {summary.window_hours} Hours]: {trades_str}, {pnl_str} P&L - {summary.markets_flagged_count} MARKET(S) FLAGGED"
        else:
            subject = f"[Dora Bot Summary Last {summary.window_hours} Hours]: {trades_str}, {pnl_str} P&L"
        logger.info(f"  Subject: {subject}")

        # Build email bodies
        logger.info("Building email bodies...")
        html_body = self.build_html_report(summary, environment)
        text_body = self.build_text_report(summary, environment)
        logger.info(f"  HTML body length: {len(html_body)} chars")
        logger.info(f"  Text body length: {len(text_body)} chars")

        try:
            logger.info("Calling SES send_email...")
            response = self.ses.send_email(
                Source=self.sender_email,
                Destination={
                    'ToAddresses': [recipient]
                },
                Message={
                    'Subject': {
                        'Data': subject,
                        'Charset': 'UTF-8'
                    },
                    'Body': {
                        'Text': {
                            'Data': text_body,
                            'Charset': 'UTF-8'
                        },
                        'Html': {
                            'Data': html_body,
                            'Charset': 'UTF-8'
                        }
                    }
                }
            )
            logger.info(f"Email sent successfully. Message ID: {response['MessageId']}")
            logger.info(f"Full SES response: {response}")
            return True
        except self.ses.exceptions.MessageRejected as e:
            logger.error(f"SES MessageRejected: {e}")
            return False
        except self.ses.exceptions.MailFromDomainNotVerifiedException as e:
            logger.error(f"SES MailFromDomainNotVerified: {e}")
            return False
        except self.ses.exceptions.ConfigurationSetDoesNotExistException as e:
            logger.error(f"SES ConfigurationSetDoesNotExist: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def send_market_proposals_email(
        self,
        proposals: list,
        proposal_id: str,
        review_url: str,
        approve_all_url: str,
        recipient: str,
        environment: str = "prod"
    ) -> bool:
        """Send market proposals email with approval links.

        Args:
            proposals: List of proposal dicts
            proposal_id: UUID of the proposal batch
            review_url: Signed URL for reviewing proposals
            approve_all_url: Signed URL for approving all
            recipient: Email recipient
            environment: 'demo' or 'prod'

        Returns:
            True if email sent successfully
        """
        # Count proposals by source and action
        update_proposals = [p for p in proposals if p['proposal_source'] == 'market_update']
        screener_proposals = [p for p in proposals if p['proposal_source'] == 'market_screener']
        all_proposals = proposals

        # Count by action
        action_counts = {}
        for p in update_proposals:
            action = p['action']
            action_counts[action] = action_counts.get(action, 0) + 1

        # Calculate summary statistics
        total_pnl = sum(p.get('metadata', {}).get('pnl_24h', 0) or 0 for p in all_proposals)
        total_fills = sum(p.get('metadata', {}).get('fill_count', 0) or 0 for p in all_proposals)
        markets_with_fills = sum(1 for p in all_proposals if (p.get('metadata', {}).get('fill_count', 0) or 0) > 0)
        markets_with_fills_pct = (markets_with_fills / len(all_proposals) * 100) if all_proposals else 0
        total_position = sum(abs(p.get('metadata', {}).get('position_qty', 0) or 0) for p in all_proposals)
        total_current_quote_size = sum(p.get('current_config', {}).get('quote_size', 0) or 0 for p in all_proposals)
        total_proposed_quote_size = sum(
            p.get('proposed_changes', {}).get('quote_size') or p.get('current_config', {}).get('quote_size', 0) or 0
            for p in all_proposals
        )

        # Get top 10 positive and negative P&Ls
        proposals_with_pnl = [(p, p.get('metadata', {}).get('pnl_24h', 0) or 0) for p in all_proposals]
        proposals_with_pnl.sort(key=lambda x: x[1], reverse=True)
        top_10_positive = [(p, pnl) for p, pnl in proposals_with_pnl if pnl > 0][:10]
        top_10_negative = [(p, pnl) for p, pnl in reversed(proposals_with_pnl) if pnl < 0][:10]

        subject = f"[DORA {environment.upper()}] Market Management Proposals - {len(proposals)} Total | P&L: ${total_pnl:+,.2f}"

        # Build HTML email body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; }}
                .section-header {{
                    background-color: #2196F3;
                    color: white;
                    padding: 10px;
                    margin: 20px 0 10px 0;
                }}
                .proposal-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                .proposal-table th, .proposal-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .proposal-table th {{ background-color: #4CAF50; color: white; }}
                .button {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 15px 32px;
                    text-decoration: none;
                    display: inline-block;
                    margin: 10px 5px;
                    border-radius: 4px;
                }}
                .expiry-notice {{
                    background-color: #fff3cd;
                    color: #856404;
                    padding: 10px;
                    border-left: 4px solid #ffc107;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DORA Market Management Proposals</h1>
                <p>Environment: {environment.upper()} | Proposal ID: {proposal_id}</p>
            </div>

            <div class="summary">
                <h2>Summary Statistics</h2>
                <table style="width: 100%; margin: 10px 0;">
                    <tr>
                        <td><strong>Total Proposals:</strong></td>
                        <td>{len(proposals)}</td>
                        <td><strong>Market Updates:</strong></td>
                        <td>{len(update_proposals)}</td>
                        <td><strong>New Candidates:</strong></td>
                        <td>{len(screener_proposals)}</td>
                    </tr>
                    <tr>
                        <td><strong>Total P&L (24h):</strong></td>
                        <td style="color: {'green' if total_pnl >= 0 else 'red'}; font-weight: bold;">${total_pnl:+,.2f}</td>
                        <td><strong>Total Fills:</strong></td>
                        <td>{total_fills:,}</td>
                        <td><strong>Markets with Fills:</strong></td>
                        <td>{markets_with_fills} ({markets_with_fills_pct:.0f}%)</td>
                    </tr>
                    <tr>
                        <td><strong>Total Position:</strong></td>
                        <td>{total_position:,}</td>
                        <td><strong>Current Quote Size:</strong></td>
                        <td>{total_current_quote_size:,}</td>
                        <td><strong>Proposed Quote Size:</strong></td>
                        <td>{total_proposed_quote_size:,}</td>
                    </tr>
                </table>
        """

        # Add action breakdown
        if action_counts:
            html_body += "<h3>Action Breakdown:</h3><ul>"
            for action, count in sorted(action_counts.items()):
                html_body += f"<li>{action.replace('_', ' ').title()}: {count}</li>"
            html_body += "</ul>"

        html_body += """
            </div>

            <div class="expiry-notice">
                ⏰ <strong>Important:</strong> This approval link expires in 12 hours.
            </div>
        """

        # Top 10 Positive P&Ls
        if top_10_positive:
            html_body += """
            <div class="section-header" style="background-color: #28a745;">
                <h2>📈 Top 10 Positive P&Ls</h2>
            </div>
            <table class="proposal-table">
                <thead>
                    <tr>
                        <th>Market ID</th>
                        <th>Action</th>
                        <th>P&L (24h)</th>
                        <th>Fills</th>
                        <th>Position</th>
                    </tr>
                </thead>
                <tbody>
            """
            for p, pnl in top_10_positive:
                fill_count = p.get('metadata', {}).get('fill_count', 0) or 0
                position_qty = p.get('metadata', {}).get('position_qty', 0) or 0
                html_body += f"""
                    <tr>
                        <td>{p['market_id']}</td>
                        <td>{p.get('action', 'N/A')}</td>
                        <td style="color: green; font-weight: bold;">${pnl:+,.2f}</td>
                        <td>{fill_count}</td>
                        <td>{self.format_position_qty(position_qty)}</td>
                    </tr>
                """
            html_body += "</tbody></table>"

        # Top 10 Negative P&Ls
        if top_10_negative:
            html_body += """
            <div class="section-header" style="background-color: #dc3545;">
                <h2>📉 Top 10 Negative P&Ls</h2>
            </div>
            <table class="proposal-table">
                <thead>
                    <tr>
                        <th>Market ID</th>
                        <th>Action</th>
                        <th>P&L (24h)</th>
                        <th>Fills</th>
                        <th>Position</th>
                    </tr>
                </thead>
                <tbody>
            """
            for p, pnl in top_10_negative:
                fill_count = p.get('metadata', {}).get('fill_count', 0) or 0
                position_qty = p.get('metadata', {}).get('position_qty', 0) or 0
                html_body += f"""
                    <tr>
                        <td>{p['market_id']}</td>
                        <td>{p.get('action', 'N/A')}</td>
                        <td style="color: red; font-weight: bold;">${pnl:+,.2f}</td>
                        <td>{fill_count}</td>
                        <td>{self.format_position_qty(position_qty)}</td>
                    </tr>
                """
            html_body += "</tbody></table>"

        # New Candidates Section
        if screener_proposals:
            html_body += """
            <div class="section-header">
                <h2>🆕 New Market Candidates (From Screener)</h2>
            </div>
            <table class="proposal-table">
                <thead>
                    <tr>
                        <th>Market ID</th>
                        <th>Title</th>
                        <th>Volume (24h)</th>
                        <th>Bid/Ask</th>
                        <th>Quote Size</th>
                    </tr>
                </thead>
                <tbody>
            """

            for p in screener_proposals[:20]:  # Limit to 20 for email size
                metadata = p.get('metadata', {})
                volume = metadata.get('volume_24h', 0)
                yes_bid = metadata.get('yes_bid', 0)
                yes_ask = metadata.get('yes_ask', 0)
                title = metadata.get('title', '')[:50]
                quote_size = p.get('proposed_changes', {}).get('quote_size', 5)

                html_body += f"""
                    <tr>
                        <td>{p['market_id']}</td>
                        <td>{title}</td>
                        <td>{volume:,}</td>
                        <td>{yes_bid}/{yes_ask}</td>
                        <td>{quote_size}</td>
                    </tr>
                """

            html_body += "</tbody></table>"

        # Action buttons
        html_body += f"""
            <div style="text-align: center; margin: 30px 0;">
                <a href="{review_url}" class="button">Review & Approve</a>
                <a href="{approve_all_url}" class="button">Approve All</a>
            </div>

            <p style="color: #666; font-size: 12px;">
                Do not share this link with others. For security, links are single-use and expire after 12 hours.
            </p>
        </body>
        </html>
        """

        # Plain text version
        text_body = f"""
DORA Market Management Proposals
Environment: {environment.upper()}
Proposal ID: {proposal_id}

Summary:
- Total Proposals: {len(proposals)}
- Market Updates: {len(update_proposals)}
- New Candidates: {len(screener_proposals)}

Review URL: {review_url}
Approve All URL: {approve_all_url}

This link expires in 12 hours.
        """

        logger.info(f"Sending market proposals email to {recipient}")
        logger.info(f"  Total proposals: {len(proposals)}")
        logger.info(f"  Update proposals: {len(update_proposals)}")
        logger.info(f"  Screener proposals: {len(screener_proposals)}")

        try:
            response = self.ses.send_email(
                Source=self.sender_email,
                Destination={'ToAddresses': [recipient]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {
                        'Text': {'Data': text_body, 'Charset': 'UTF-8'},
                        'Html': {'Data': html_body, 'Charset': 'UTF-8'}
                    }
                }
            )
            logger.info(f"Proposal email sent successfully. Message ID: {response['MessageId']}")
            return True
        except Exception as e:
            logger.error(f"Failed to send proposal email: {e}")
            return False
