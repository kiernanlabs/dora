[
    {
        "@timestamp": "2025-12-25 17:08:34.032",
        "@message": {
            "ts": "2025-12-25T17:08:29.686934+00:00",
            "level": "ERROR",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Kalshi API Error",
            "event_type": "ERROR",
            "decision_id": "20251225-170059-62875e:KXTRUMPMEETING-27JAN01-NKJU:7",
            "market": "KXTRUMPMEETING-27JAN01-NKJU",
            "status_code": 404,
            "url": "https://api.elections.kalshi.com/trade-api/v2/portfolio/orders/9988654a-667a-487b-b73d-2a24a50052a5",
            "response_headers": {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": "73",
                "Connection": "keep-alive",
                "Date": "Thu, 25 Dec 2025 17:08:29 GMT",
                "x-content-type-options": "nosniff",
                "content-security-policy": "default-src 'none';",
                "strict-transport-security": "max-age=31536000; includeSubDomains",
                "X-Cache": "Error from cloudfront",
                "Via": "1.1 7b0a3da03f41361eb04138752c927d30.cloudfront.net (CloudFront)",
                "X-Amz-Cf-Pop": "IAD61-P7",
                "X-Amz-Cf-Id": "I55WZNlhqR-hn3Q7A0p0a5SR4Fh6l6XTrD48sRpRuabb3a58d6LaUw=="
            },
            "response_body": "{\"error\":{\"code\":\"not_found\",\"message\":\"not found\",\"service\":\"exchange\"}}",
            "logger": "dora_bot.kalshi_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:07:36.032",
        "@message": {
            "ts": "2025-12-25T17:07:31.666675+00:00",
            "level": "ERROR",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Kalshi API Error",
            "event_type": "ERROR",
            "decision_id": "20251225-170059-62875e:KXIPO-26-DATABRICKS:6",
            "market": "KXIPO-26-DATABRICKS",
            "status_code": 404,
            "url": "https://api.elections.kalshi.com/trade-api/v2/portfolio/orders/2640b185-12d4-417c-8da2-0035faa67df3",
            "response_headers": {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": "73",
                "Connection": "keep-alive",
                "Date": "Thu, 25 Dec 2025 17:07:31 GMT",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none';",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "X-Cache": "Error from cloudfront",
                "Via": "1.1 05a63efd5ebb043211ccf075f3a8aa68.cloudfront.net (CloudFront)",
                "X-Amz-Cf-Pop": "IAD61-P10",
                "X-Amz-Cf-Id": "gki3zBc_LmnEvH7J-zS_eSy_VMhgp1F6UcSlg5dRIbuzEwgwfwY6xg=="
            },
            "response_body": "{\"error\":{\"code\":\"not_found\",\"message\":\"not found\",\"service\":\"exchange\"}}",
            "logger": "dora_bot.kalshi_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:06:07.032",
        "@message": {
            "ts": "2025-12-25T17:06:01.903202+00:00",
            "level": "ERROR",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Kalshi API Error",
            "event_type": "ERROR",
            "decision_id": "20251225-170059-62875e:KXIPO-26-DATABRICKS:4",
            "market": "KXIPO-26-DATABRICKS",
            "status_code": 404,
            "url": "https://api.elections.kalshi.com/trade-api/v2/portfolio/orders/020c0c3d-74cb-4c4f-8647-423b18fd2d17",
            "response_headers": {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": "73",
                "Connection": "keep-alive",
                "Date": "Thu, 25 Dec 2025 17:06:01 GMT",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none';",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "X-Cache": "Error from cloudfront",
                "Via": "1.1 dcdda2de0f9d7443c4c888a61edd2c22.cloudfront.net (CloudFront)",
                "X-Amz-Cf-Pop": "IAD61-P4",
                "X-Amz-Cf-Id": "HOkoGI1Nb-_bLB_GiFGaVQcMiz0bHzC1YUyVQ5GoacQ7IihJrpNsMQ=="
            },
            "response_body": "{\"error\":{\"code\":\"not_found\",\"message\":\"not found\",\"service\":\"exchange\"}}",
            "logger": "dora_bot.kalshi_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:04:59.718",
        "@message": {
            "ts": "2025-12-25T17:04:59.518582+00:00",
            "level": "INFO",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Order already cancelled/filled",
            "event_type": "ORDER_RESULT",
            "decision_id": "20251225-170059-62875e:KXIPO-26-DATABRICKS:2",
            "market": "KXIPO-26-DATABRICKS",
            "order_id": "f3344bb9-3405-4fd3-b15a-22aecf9dc202",
            "status": "ALREADY_GONE",
            "latency_ms": 133,
            "client_order_id": null,
            "logger": "dora_bot.exchange_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:04:38.032",
        "@message": {
            "ts": "2025-12-25T17:04:32.927540+00:00",
            "level": "ERROR",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Kalshi API Error",
            "event_type": "ERROR",
            "decision_id": "20251225-170059-62875e:KXIPO-26-DATABRICKS:2",
            "market": "KXIPO-26-DATABRICKS",
            "status_code": 404,
            "url": "https://api.elections.kalshi.com/trade-api/v2/portfolio/orders/f3344bb9-3405-4fd3-b15a-22aecf9dc202",
            "response_headers": {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": "73",
                "Connection": "keep-alive",
                "Date": "Thu, 25 Dec 2025 17:04:32 GMT",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none';",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "X-Cache": "Error from cloudfront",
                "Via": "1.1 554a247e2bb62ed2a3603decd985d5d6.cloudfront.net (CloudFront)",
                "X-Amz-Cf-Pop": "IAD61-P6",
                "X-Amz-Cf-Id": "tRcIP2vrOZLAr5HBwqSFu0kPKxXrULcgQCJ11yjvjFWeSuK6Tf2Rjw=="
            },
            "response_body": "{\"error\":{\"code\":\"not_found\",\"message\":\"not found\",\"service\":\"exchange\"}}",
            "logger": "dora_bot.kalshi_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:04:33.149",
        "@message": {
            "ts": "2025-12-25T17:04:32.793978+00:00",
            "level": "INFO",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Cancelling order",
            "event_type": "ORDER_CANCEL",
            "decision_id": "20251225-170059-62875e:KXIPO-26-DATABRICKS:2",
            "market": "KXIPO-26-DATABRICKS",
            "order_id": "f3344bb9-3405-4fd3-b15a-22aecf9dc202",
            "client_order_id": null,
            "logger": "dora_bot.exchange_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:04:09.033",
        "@message": {
            "ts": "2025-12-25T17:04:04.275517+00:00",
            "level": "ERROR",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Kalshi API Error",
            "event_type": "ERROR",
            "decision_id": "20251225-170059-62875e:KXIPOOPENAI-26AUG01:2",
            "market": "KXIPOOPENAI-26AUG01",
            "status_code": 404,
            "url": "https://api.elections.kalshi.com/trade-api/v2/portfolio/orders/6a205d45-0967-4389-aeba-d6b80618ae77",
            "response_headers": {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": "73",
                "Connection": "keep-alive",
                "Date": "Thu, 25 Dec 2025 17:04:04 GMT",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none';",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "X-Cache": "Error from cloudfront",
                "Via": "1.1 c2a3a87aaabb428d0166775a267ba578.cloudfront.net (CloudFront)",
                "X-Amz-Cf-Pop": "IAD61-P11",
                "X-Amz-Cf-Id": "VoZu1I8LGD0GWFU_qTdUsHQyCt_2VvVCB9YU2t5118L3ghW1VlMuOg=="
            },
            "response_body": "{\"error\":{\"code\":\"not_found\",\"message\":\"not found\",\"service\":\"exchange\"}}",
            "logger": "dora_bot.kalshi_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:03:42.032",
        "@message": {
            "ts": "2025-12-25T17:03:37.378505+00:00",
            "level": "ERROR",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Kalshi API Error",
            "event_type": "ERROR",
            "decision_id": "20251225-170059-62875e:KXIPOOPENAI-26AUG01:2",
            "market": "KXIPOOPENAI-26AUG01",
            "status_code": 404,
            "url": "https://api.elections.kalshi.com/trade-api/v2/portfolio/orders/3786e4e8-8011-45ea-bb95-d67c65b10394",
            "response_headers": {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": "73",
                "Connection": "keep-alive",
                "Date": "Thu, 25 Dec 2025 17:03:37 GMT",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none';",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "X-Cache": "Error from cloudfront",
                "Via": "1.1 19a1f4ad0433ae2397d314489416121e.cloudfront.net (CloudFront)",
                "X-Amz-Cf-Pop": "IAD61-P11",
                "X-Amz-Cf-Id": "__wnmPF6yFORm0kmc0eDY7XfHICI_9Lqtuwcr8637x_gCgEW8pc8eA=="
            },
            "response_body": "{\"error\":{\"code\":\"not_found\",\"message\":\"not found\",\"service\":\"exchange\"}}",
            "logger": "dora_bot.kalshi_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:03:07.432",
        "@message": {
            "ts": "2025-12-25T17:03:07.210057+00:00",
            "level": "INFO",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Cancelling order",
            "event_type": "ORDER_CANCEL",
            "decision_id": "20251225-170059-62875e:KXIPO-26-DATABRICKS:1",
            "market": "KXIPO-26-DATABRICKS",
            "order_id": "f3344bb9-3405-4fd3-b15a-22aecf9dc202",
            "client_order_id": "630ebad27a2b1a70",
            "logger": "dora_bot.exchange_client"
        }
    },
    {
        "@timestamp": "2025-12-25 17:03:07.432",
        "@message": {
            "ts": "2025-12-25T17:03:07.346098+00:00",
            "level": "INFO",
            "service": "dora-bot",
            "env": "prod",
            "bot_version": "unknown",
            "bot_run_id": "20251225-170059-62875e",
            "message": "Order cancelled",
            "event_type": "ORDER_RESULT",
            "decision_id": "20251225-170059-62875e:KXIPO-26-DATABRICKS:1",
            "market": "KXIPO-26-DATABRICKS",
            "order_id": "f3344bb9-3405-4fd3-b15a-22aecf9dc202",
            "status": "CANCELLED",
            "latency_ms": 135,
            "client_order_id": "630ebad27a2b1a70",
            "logger": "dora_bot.exchange_client"
        }
    }
]