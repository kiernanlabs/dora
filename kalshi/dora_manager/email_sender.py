"""
SES Email Sender for Dora Manager Reports
"""
import boto3
from typing import Optional
import logging
from .calculator import TradingSummary, MarketSummary

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
