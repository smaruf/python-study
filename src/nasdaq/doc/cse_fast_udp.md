# Chittagong Stock Exchange Market Data Feed UDP FIX/FAST Specification

**Version:** 1.07  
**Release Date:** 5th Sep 2011  
**Number of Pages:** 61 (Including Cover Page)

---

## Document Information

**Millennium Information Technologies**  
1, Millennium Drive, Malabe, Sri Lanka  
Website: www.millenniumit.com  
Tel: +94 11 241 6000  
Fax: +94 11 241 3227  
Email: info@millenniumit.com

---

## Confidentiality Notice

Information contained in this specification is proprietary to Millennium IT Software (Private) Limited and is confidential. It shall not be reproduced or disclosed in whole or in part to any party (other than to any individual who has a need to peruse the content of this specification in connection with the purpose for which it is submitted) or used for any purpose other than the purpose for which it is submitted, without the written approval of Millennium IT Software (Private) Limited.

**Copyright © 2010, Millennium IT (Software) Limited**

---

## Table of Contents

1. [DOCUMENT CONTROL](#1-document-control)
   - 1.1 [Table of Contents](#11-table-of-contents)
   - 1.2 [Document Information](#12-document-information)
   - 1.3 [Revision History](#13-revision-history)
   - 1.4 [References](#14-references)
   - 1.5 [Definitions, Acronyms and Abbreviations](#15-definitions-acronyms-and-abbreviations)

2. [OVERVIEW](#2-overview)
   - 2.1 [Hours of Operation](#21-hours-of-operation)
   - 2.2 [Support](#22-support)

3. [SERVICE DESCRIPTION](#3-service-description)
   - 3.1 [System Architecture](#31-system-architecture)
     - 3.1.1 [Real-Time Channel](#311-real-time-channel)
     - 3.1.2 [Snapshot Channel](#312-snapshot-channel)
     - 3.1.3 [Replay Channel](#313-replay-channel)
   - 3.2 [Message Overview](#32-message-overview)
   - 3.3 [Overview of a Trading Day](#33-overview-of-a-trading-day)
     - 3.3.1 [Trading on the Order Book](#331-trading-on-the-order-book)
     - 3.3.2 [Trade Reporting](#332-trade-reporting)
     - 3.3.3 [Trading Halt](#333-trading-halt)
     - 3.3.4 [Pause](#334-pause)
     - 3.3.5 [Instrument Suspension](#335-instrument-suspension)
     - 3.3.6 [Intra-Day Trading Session Updates](#336-intra-day-trading-session-updates)
     - 3.3.7 [New Instruments](#337-new-instruments)
   - 3.4 [Order Book Management (Price Depth)](#34-order-book-management-price-depth)
     - 3.4.1 [Incremental Refresh](#341-incremental-refresh)
     - 3.4.2 [Snapshot](#342-snapshot)
     - 3.4.3 [Market Orders](#343-market-orders)
   - 3.5 [Order Book Management (Order Depth)](#35-order-book-management-order-depth)
     - 3.5.1 [Incremental Refresh](#351-incremental-refresh)
     - 3.5.2 [Snapshot](#352-snapshot)
     - 3.5.3 [Market Orders](#353-market-orders)
     - 3.5.4 [Attributed Orders](#354-attributed-orders)
   - 3.6 [Time and Sales](#36-time-and-sales)
     - 3.6.1 [Auction Trades](#361-auction-trades)
     - 3.6.2 [Pre-Negotiated Trades](#362-pre-negotiated-trades)
     - 3.6.3 [Trade Cancellation and Corrections](#363-trade-cancellation-and-corrections)
   - 3.7 [Indicative Auction Information](#37-indicative-auction-information)
   - 3.8 [Statistics](#38-statistics)
     - 3.8.1 [Book-Level Statistics](#381-book-level-statistics)
     - 3.8.2 [Market and Sector Statistics](#382-market-and-sector-statistics)
     - 3.8.3 [Mode of Dissemination](#383-mode-of-dissemination)
   - 3.9 [Indices](#39-indices)
   - 3.10 [Quotation Conventions](#310-quotation-conventions)
   - 3.11 [Announcements](#311-announcements)

4. [CONNECTIVITY](#4-connectivity)
   - 4.1 [Transmission Standards](#41-transmission-standards)
     - 4.1.1 [Multicast Channels](#411-multicast-channels)
     - 4.1.2 [Point-to-Point Channels](#412-point-to-point-channels)
   - 4.2 [Application IDs](#42-application-ids)
     - 4.2.1 [Server](#421-server)
     - 4.2.2 [Clients](#422-clients)
   - 4.3 [Production IP Addresses and Ports](#43-production-ip-addresses-and-ports)
     - 4.3.1 [Main Site](#431-main-site)
     - 4.3.2 [Backup Site](#432-backup-site)
   - 4.4 [Bandwidth](#44-bandwidth)

5. [RECOVERY](#5-recovery)
   - 5.1 [Recipient Failures](#51-recipient-failures)
     - 5.1.1 [Snapshot Channel](#511-snapshot-channel)
     - 5.1.2 [Replay Channel](#512-replay-channel)
   - 5.2 [Failures at The Chittagong Stock Exchange](#52-failures-at-the-chittagong-stock-exchange)
     - 5.2.1 [Snapshots on the Real-Time Channel](#521-snapshots-on-the-real-time-channel)
     - 5.2.2 [Resetting Sequence Numbers](#522-resetting-sequence-numbers)

6. [MESSAGE FORMATS AND TEMPLATES](#6-message-formats-and-templates)
   - 6.1 [Variations from the FIX Protocol](#61-variations-from-the-fix-protocol)
   - 6.2 [Header](#62-header)
     - 6.2.1 [FIX Message](#621-fix-message)
     - 6.2.2 [FAST Template](#622-fast-template)
   - 6.3 [Administrative Messages](#63-administrative-messages)
     - 6.3.1 [Logon](#631-logon)
     - 6.3.2 [Logout](#632-logout)
     - 6.3.3 [Heartbeat](#633-heartbeat)
   - 6.4 [Application Messages (Client-Initiated)](#64-application-messages-client-initiated)
     - 6.4.1 [Security Definition Request](#641-security-definition-request)
     - 6.4.2 [Market Data Request](#642-market-data-request)
     - 6.4.3 [Application Message Request](#643-application-message-request)
   - 6.5 [Application Messages (Server-Initiated)](#65-application-messages-server-initiated)
     - 6.5.1 [Security Definition](#651-security-definition)
     - 6.5.2 [Security Status](#652-security-status)
     - 6.5.3 [Market Data Snapshot (Full Refresh)](#653-market-data-snapshot-full-refresh)
     - 6.5.4 [Market Data Incremental Refresh](#654-market-data-incremental-refresh)
     - 6.5.5 [News](#655-news)
     - 6.5.6 [Market Data Request Reject](#656-market-data-request-reject)
     - 6.5.7 [Business Message Reject](#657-business-message-reject)
     - 6.5.8 [Application Message Request Ack](#658-application-message-request-ack)
     - 6.5.9 [Application Message Report](#659-application-message-report)

7. [INSTRUMENT CLASSIFICATION](#7-instrument-classification)
   - 7.1 [Segment](#71-segment)
   - 7.2 [CFI Codes](#72-cfi-codes)
   - 7.3 [Security Types](#73-security-types)

8. [OFF-BOOK TRADE TYPES](#8-off-book-trade-types)

9. [TRADING HALT REASON CODES](#9-trading-halt-reason-codes)

10. [REJECT CODES](#10-reject-codes)
    - 10.1 [Market Data Request Reject](#101-market-data-request-reject)
    - 10.2 [Business Message Reject](#102-business-message-reject)

---

## 1. Document Control

### 1.1 Table of Contents
*(See above)*

### 1.2 Document Information

| Field | Value |
|-------|-------|
| Drafted By | Niren Neydorff |
| Status | Draft |
| Version | 1.07 |
| Release Date | 5th Sep 2011 |

### 1.3 Revision History

| Date | Version | Sections | Description |
|------|---------|----------|-------------|
| 20 Oct 10 | 1.00 | - | Initial Draft |
| 01 Feb 11 | 1.01 | - | Overall review including Chittagong Stock Exchange specific requirements |
| 10 Feb 11 | 1.02 | 3.1.0, 6.5.4 | Market and Sector level statistics introduced to the Market Data Incremental Refresh message. |
| | | 3.3.1, 6.5.2.1, 6.5.3.1 | Opening and Closing Price Publication sessions introduced to the Trading day overview as well as the Security Status and Market Data Snapshot messages. |
| | | 6.4.2.1 | The MDSubBookType (1173) field of the Market Data Request message was updated with the relevant order book data. |
| 22 Feb 11 | 1.03 | 3.8.1, 3.8.2, 3.8.3 | Re-formatted the statistics description. |
| | | 3.9, 6.5.4.1 | Closing Index and Previous Closing Index values introduced to the Indices overview as well as the Market Data Incremental Refresh message. |
| 25 Mar 11 | 1.04 | | Overall review. |
| 05 May 11 | 1.05 | | Introduced Buy and Sell Order VWAP as additional Book Level statistics. Added additional Closing price indications. Added the relevant segments (categories) for instruments. |
| 20-06-2011 | 1.06 | 6.5.3.1 | Buy Order Qty, Sell Order Qty added to snapshot message |
| | | 6.5.4.1 | Buy Order Qty, Sell Order Qty added to incremental refresh message |
| 05-09-2011 | 1.07 | 3.8.1, 6.5.3.1 | Market data and order book updates. Field 269 – new value "Previous Close Price" is added. Field 270 – Fields Buy Order Qty (u), Sell Order Qty (v) are added |
| | | 6.5.4.1 | Field 269 – new value "Previous Close Price" is added. Field 270 – Fields Buy Order Qty (u), Sell Order Qty (v) are added |

### 1.4 References

- FAST 1.1 Session Control Protocol Specification
- FIX 5.0 (Service Pack 2) Specification

### 1.5 Definitions, Acronyms and Abbreviations

| Term | Definition |
|------|------------|
| Client | A recipient connected to the Snapshot or Replay channel of the market data feed. |
| FAST | Version 1.1 of the Session Control Protocol of the FIX Adapted for STreaming specification. |
| FIX | Version 5.0 (Service Pack 2) of the Financial Information Exchange Protocol. |
| Off-Book Trade | A privately negotiated trade that is reported to The Chittagong Stock Exchange. |
| Orders | Executable interest in the order book. |
| Pre-Auction | The trading session immediately prior to an auction (e.g. opening). During this session orders are accumulated for execution in the auction and information on the indicative auction price and associated imbalance is disseminated at a regular interval. |
| Recipient | A subscriber to the market data feed. |
| Server | The market data interface of The Chittagong Stock Exchange. |
| Sub Book | Each instrument is traded across multiple separate and distinct sub books (e.g. regular, odd lot, etc.). Messages transmitted on the feed include an indication of the instrument and sub book to which it relates. |
| Trade Reporting | The reporting of an off-book trade. |
| VWAP | Volume weighted average price. |

---

## 2. Overview

The market data feed is a stream of FAST encoded FIX messages which provides the following real-time information for each instrument traded on The Chittagong Stock Exchange:

1. **Price depth information** for the order book. The feed provides information on the aggregated displayed quantity and the number of displayed orders for each of the top five price points.
2. **Order depth information** for the order book. The feed provides information on the price and displayed quantity of each order in the top five price points.
3. **Details** (e.g. price, volume, time, etc.) of on and off-book trades.
4. **Indicative auction price** and the associated trade volume and imbalance.
5. **Statistics** (e.g. high/low, volume, VWAP, etc.).
6. **Trading status**.

Each instrument is traded on a series of separate and independent sub books (e.g. regular, odd lot, etc.). The above information is disseminated per instrument and sub book combination. An update transmitted on the feed includes an indication of the instrument and sub book to which it relates.

In addition, the feed includes market and sector statistics as well as the value of each index computed by The Chittagong Stock Exchange. It also provides participants with the active instrument list of and disseminates market announcements.

The feed is a multicast service based on the technology and industry standards UDP, IPv4, FAST and FIX. The application messages are defined using the FIX 5.0 (Service Pack 2) standard and comply with the best practices outlined by the FIX Market Data Working Group. Please refer to Section 6.1 for the instances where the server varies from the FIX protocol. The data feed is transmitted in the FAST encoding method to minimize bandwidth and reduce latency and conforms to Level 1 of the FAST 1.1 specification.

### 2.1 Hours of Operation

The feed will operate from `<Start Time>` to `<End Time>` each trading day.

### 2.2 Support

`<Insert support information for recipients (e.g. contact details and hours of operation for the support desk)>`

---

## 3. Service Description

### 3.1 System Architecture

The market data feed is load balanced by market data group. While each group will contain multiple instruments, each instrument is assigned to just one market data group. Although the group an instrument is assigned to may change from day to day, it will not change within a day. Market data for all sub books (e.g. regular, odd lot, etc.) for a particular instrument are transmitted from the same market data group.

Each market data group includes a multicast Real-Time channel for the dissemination of market data. Two TCP recovery channels are available per market data group; Snapshot and Replay.

While a recipient may connect to the Replay channel to recover from a small data loss, it should use the Snapshot channel after a large data loss (i.e. late joiner or major outage).

```
Market Data Group (Main Site)
┌─────────────────────────────┐    ┌─────────────────────────────┐
│     Real-Time Channel A     │    │     Real-Time Channel B     │
│          (UDP)              │    │          (UDP)              │
│                             │    │                             │
│ • Instruments               │    │ • Instruments               │
│ • Order Book Updates        │    │ • Order Book Updates        │
│ • Trades                    │    │ • Trades                    │
│ • Indicative Auction Info   │    │ • Indicative Auction Info   │
│ • Statistics Updates        │    │ • Statistics Updates        │
│ • Trading Status            │    │ • Trading Status            │
│ • RFQs                      │    │ • RFQs                      │
│ • Announcements             │    │ • Announcements             │
└─────────────────────────────┘    └─────────────────────────────┘
                │                                     │
                │                                     │
                ▼                                     ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│     Snapshot Channel        │    │     Snapshot Channel        │
│          (TCP)              │    │          (TCP)              │
│                             │    │                             │
│ • Order Book               │    │ • Order Book               │
│ • Statistics, Trades       │    │ • Statistics, Trades       │
│ • Trading Status           │    │ • Trading Status           │
│ • Instruments              │    │ • Instruments              │
└─────────────────────────────┘    └─────────────────────────────┘

┌─────────────────────────────┐    ┌─────────────────────────────┐
│      Replay Channel         │    │      Replay Channel         │
│          (TCP)              │    │          (TCP)              │
│                             │    │                             │
│ • Missed Message Request    │    │ • Missed Message Request    │
│ • Missed Incremental Updates│    │ • Missed Incremental Updates│
└─────────────────────────────┘    └─────────────────────────────┘
                │                                     │
                └─────────────┬───────────────────────┘
                              │
                              ▼
                        Recipients
Market Data Group (Backup Site)
```

#### 3.1.1 Real-Time Channel

The Real-Time Channel is the primary means of disseminating market data. Real-time updates to instruments and all market data supported by the feed are available on this multicast channel.

The list of active instruments in the market data group is broadcast at the start of the trading day via the Security Definition message. The details of instruments created during trading hours will also be disseminated via this message. Real-time updates of the trading status of instruments will be disseminated via the Security Status message.

Real-time updates to order books, indicative auction information and statistics are published along with the details of each trade via the Market Data Incremental Refresh message. While each Market Data Incremental Refresh includes a channel specific message sequence number in the field ApplSeqNum (1181), each market data entry in the message includes an instrument specific sequence number in the field RptSeq (83). The channel and instrument level sequence numbers are to reset to 1 at the start of each day.

The server will use the Heartbeat message to exercise the communication line during periods of inactivity. A Heartbeat will be sent every 2 seconds when the Real-Time channel is inactive.

Recipients have access to two identically sequenced Real-Time feeds; one from the main site (Feed A) and one from the backup site (Feed B). It is recommended that recipients process both feeds and arbitrate between them to minimise the probability of a data loss.

#### 3.1.2 Snapshot Channel

The TCP Snapshot channel permits recipients to request a snapshot of the order book and statistics for any active instrument in the market data group as well as its current trading status. In addition, it enables recipients to request the retransmission of the trades published during the last 10 minutes on the Real-Time channel. It also enables recipients to download the list of active instruments in the market data group. This channel may be used by recipients to recover from a large-scale data loss.

All messages sent by the server are transfer encoded in terms of the FAST protocol. While all application messages sent by the server (e.g. Market Data Snapshot (Full Refresh)) are field encoded, the administrative messages it sends (e.g. Logon, Heartbeat, etc.) are not. All messages (i.e. both administrative and application) initiated by the client should be transfer encoded but not field encoded.

While a Snapshot channel is available from the backup site, it will only be activated in the unlikely event of an outage at the main site.

#### 3.1.3 Replay Channel

The TCP Replay channel permits recipients to request the retransmission of a limited number of messages already published on the Real-Time channel. This channel may be used by recipients to recover from a small data loss.

The Replay channel supports the retransmission of the last 10,000 messages published on the Real-Time channel. The channel does not support the retransmission of messages published on the Snapshot channel or from previous trading days.

All messages sent by the server are transfer encoded in terms of the FAST protocol. While all application messages sent by the server (e.g. Market Data Incremental Refresh, Security Definition, etc.) are field encoded, the administrative messages it sends (e.g. Logon, Heartbeat, etc.) are not. All messages (i.e. both administrative and application) initiated by the client should be transfer encoded but not field encoded.

While a Replay channel is available from the backup site, it will only be activated in the unlikely event of an outage at the main site.

---
