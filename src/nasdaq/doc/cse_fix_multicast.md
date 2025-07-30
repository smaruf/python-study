# Chittagong Stock Exchange Trading Gateway
## FIX 5.0 SP2 Specification

**Version:** 2.00  
**Release Date:** 6th September 2021  
**Number of Pages:** 45 (Including Cover Page)

---

### Document Information

**Publisher:** Millennium Information Technologies  
**Address:** 1, Millennium Drive, Malabe, Sri Lanka  
**Website:** www.millenniumit.com  
**Tel:** +94 11 241 6000  
**Fax:** +94 11 241 3227  
**Email:** info@millenniumit.com  

---

### Copyright Notice

Information contained in this specification is proprietary to Millennium IT Software (Private) Limited and is confidential. It shall not be reproduced or disclosed in whole or in part to any party (other than to any individual who has a need to peruse the content of this specification in connection with the purpose for which it is submitted) or used for any purpose other than the purpose for which it is submitted, without the written approval of Millennium IT Software (Private) Limited.

**Copyright © 2010, Millennium IT (Software) Limited**

---

## Table of Contents

1. [Document Control](#1-document-control)
   - 1.1 [Table of Contents](#11-table-of-contents)
   - 1.2 [Document Information](#12-document-information)
   - 1.3 [Revision History](#13-revision-history)
   - 1.4 [References](#14-references)
   - 1.5 [Definitions, Acronyms and Abbreviations](#15-definitions-acronyms-and-abbreviations)
2. [Overview](#2-overview)
   - 2.1 [Hours of Operation](#21-hours-of-operation)
   - 2.2 [Support](#22-support)
3. [Service Description](#3-service-description)
   - 3.1 [Order Handling](#31-order-handling)
   - 3.2 [Odd Lot Order Book](#32-odd-lot-order-book)
   - 3.3 [Bulk Order Book](#33-bulk-order-book)
   - 3.4 [Foreign Order Book](#34-foreign-order-book)
   - 3.5 [Buy back and Issuing Auction Order Book](#35-buy-back-and-issuing-auction-order-book)
   - 3.6 [Short Sales](#36-short-sales)
   - 3.7 [Closing Price trading session](#37-closing-price-trading-session)
   - 3.8 [Party Identification](#38-party-identification)
   - 3.9 [Quotation Conventions](#39-quotation-conventions)
   - 3.10 [Market Operations](#310-market-operations)
   - 3.11 [Timestamps and Dates](#311-timestamps-and-dates)
4. [Connectivity](#4-connectivity)
5. [FIX Connections and Sessions](#5-fix-connections-and-sessions)
6. [Recovery](#6-recovery)
7. [Message Formats](#7-message-formats)
8. [Segments](#8-segments)
9. [Reject Codes](#9-reject-codes)
10. [Process Flows](#10-process-flows)

---

## 1. Document Control

### 1.2 Document Information

| Field | Value |
|-------|-------|
| **Drafted By** | Niren Neydorff, Vickmal Meemaduma |
| **Status** | Draft |
| **Version** | 2.00 |
| **Release Date** | 6th September 2021 |

### 1.3 Revision History

| Date | Version | Sections | Description |
|------|---------|----------|-------------|
| 20 Oct 10 | 1.00 | - | Initial Draft |
| 27 Jan 11 | 1.01 | - | Overall reviewed version with Chittagong Stock Exchange specific requirements |
| 09 Feb 11 | 1.02 | - | Added ability to submit orders for the closing price trading session. Also filtered out sections not required |
| 25 Mar 11 | 1.03 | - | Overall reviewed version. Fields to denote stop and stop limit order election introduced |
| 05 May 11 | 1.04 | - | Changed free text field from Text (5) and ClientText (31000) to OrderSource (30004) across all relevant messages. Added the relevant segments (categories) for instruments |
| 06 Sep 21 | 2.00 | 7.5.5 | Introduction of Yield (236) and AccruedInterestAmt (159) fields to Execution Report for Fixed Income implementation |

### 1.4 References

- FIXT 1.1 Specification
- FIX 5.0 (Service Pack 2) Specification

### 1.5 Definitions, Acronyms and Abbreviations

| Term | Definition |
|------|------------|
| **Client** | A participant or service bureau connected to the trading gateway |
| **FIX** | Version 5.0 (Service Pack 2) of the Financial Information Exchange Protocol |
| **FIX Connection** | A bi-directional stream of ordered messages between the client and server within a particular login. A FIX connection ends when the client logs out or if the TCP/IP connection is terminated |
| **FIX Session** | A bi-directional stream of ordered messages between the client and server within a continuous sequence number series. A single FIX session can exist across multiple FIX connections |
| **FIXT** | Version 1.1 of the Financial Information Exchange Session Protocol |
| **Interest** | Executable interest |
| **Order Book** | Each instrument is traded across multiple separate and distinct order books (e.g. regular, odd lot, etc.). Each order submitted by a client should include an indication of the instrument and order book to which it relates |
| **Server** | The trading gateway of The Chittagong Stock Exchange |
| **Trading Mnemonic** | Each order request must be submitted under a particular trading mnemonic. Trading privileges are assigned to participants at the level of their trading mnemonics |

---

## 2. Overview

The Chittagong Stock Exchange offers a trading gateway which will allow participants and service bureaus to send and manage their trading interest. The interface enables clients to perform the activities outlined below.

### Order Handling

1. Submit an order
2. Cancel an order
3. Mass cancel orders
4. Cancel/replace an order

The interface is a point-to-point service based on the technology and industry standards TCP/IP, FIXT and FIX. The session and application event models and messages are based on versions 1.1 and 5.0 (Service Pack 2) of the FIXT and FIX protocols respectively. Please refer to Section 7.2 for the instances where the server varies from the FIX protocol.

> **Note:** The encryption of messages between the client and server is not supported.

### 2.1 Hours of Operation

The server will operate from `<Start Time>` to `<End Time>` each trading day.

### 2.2 Support

`<Insert support information for clients (e.g. contact details and hours of operation for the support desk)>`

---

## 3. Service Description

### 3.1 Order Handling

#### 3.1.1 Order Types

Clients may submit the order types outlined below via the New Order – Single message.

| Order Type | Description | Relevant FIX Tags |
|------------|-------------|-------------------|
| **Market** | An order that will execute at the best available prices until it is fully filled. Any remainder will be cancelled | `OrderType (40) = 1` |
| **Market To Limit** | An order that will execute at the best available prices until it is fully filled. Any remainder will be added to the order book (based on the TimeInForce (59) specified) as a limit order priced at the last traded price | `OrderType (40) = K` |
| **Limit** | An order that will execute at or better than the specified price. The remainder, if any, is added to the order book or expired in terms of its TimeInForce (59) | `OrderType (40) = 2`<br>`Price (44)` |
| **Stop** | A market order that remains inactive until the market reaches a specified stop price | `OrderType (40) = 3`<br>`StopPx (99)` |
| **Stop Limit** | A limit order that remains inactive until the market reaches a specified stop price | `OrderType (40) = 4`<br>`StopPx (99)`<br>`Price (44)` |
| **Iceberg** | An order that contains a disclosed quantity which will be the maximum quantity displayed in the order book. Once the displayed quantity is reduced to zero, it will be replenished by the lower of the disclosed quantity and the remainder | `DisplayQty (1138)`<br>`OrderQty (38)` |
| **Reserve** | An order that contains no displayed quantity and is not displayed in the order book | `DisplayQty (1138) = 0` or `DisplayMethod (1084) = 4` |
| **Minimum Fill** | An order that contains a minimum quantity. If this quantity cannot be filled on receipt the order will immediately expire. If the minimum quantity is filled, the remainder, if any, is added to the order book as a regular order or expired in terms of its TimeInForce (59) | `MinQty (110)` |

#### Time In Force Options

| Order Type | Description | Relevant FIX Tags |
|------------|-------------|-------------------|
| **Day** | An order that will expire at the end of the day | `TimeInForce (59) = 0` |
| **Immediate or Cancel (IOC)** | An order that will be executed on receipt and the remainder, if any, immediately cancelled | `TimeInForce (59) = 3` |
| **Fill or Kill (FOK)** | An order that will be fully executed on receipt or immediately cancelled | `TimeInForce (59) = 4` |
| **At the Open (OPG)** | An order that may only be executed in the opening auction | `TimeInForce (59) = 2` |
| **Good Till Time (GTT)** | An order that will expire at a specified time during the current day | `TimeInForce (59) = 6`<br>`ExpireTime (126)` |
| **Good Till Date (GTD)** | An order that will expire at the end of a specified day | `TimeInForce (59) = 6`<br>`ExpireDate (432)` |
| **Good Till Cancelled (GTC)** | An order that will never expire | `TimeInForce (59) = 1` |

##### 3.1.1.1 Order Capacity

The server recognises two order capacities; agency and principal. Clients are responsible for indicating the capacity an order is submitted under. If a New Order – Single message does not contain the OrderCapacity (528) field, the server will treat the order as an agency order.

The order book is sorted in terms of price, capacity and time. Within a particular price point all agency orders will, irrespective of the time they were received, have a higher priority than principal orders.

#### 3.1.2 Order Management

##### 3.1.2.1 Cancellation

The remainder of a live order may be cancelled via the Order Cancel Request message. The server will respond with an Execution Report or Order Cancel Reject to confirm or reject the cancellation request respectively.

The client should identify the order being cancelled by either its `OrigClOrdID (41)` or `OrderID (37)`. If an Order Cancel Request contains values for both `OrigClOrdID (41)` and `OrderID (37)`, the server will only process the `OrderID (37)`. If an order submitted under a different `SenderCompID (49)` is being cancelled, the Order Cancel Request should include its `OrderID (37)`.

##### 3.1.2.2 Mass Cancellation

A client may mass cancel live orders via the Order Mass Cancel Request message. The server will respond with an Order Mass Cancel Report to indicate, via the `MassCancelResponse (531)` field, whether the request is successful or not. If the mass cancel request is processed by multiple partitions, an Order Mass Cancel Report will be transmitted for each partition.

If the mass cancel request is accepted by a partition, it will then transmit Execution Reports for each order that is cancelled and Order Cancel Rejects for each order that could not be cancelled. The `ClOrdID (11)` of all such messages will be the `ClOrdID (11)` of the Order Mass Cancel Request.

If the mass cancel request is rejected by a partition, the reason will be specified in the `MassCancelRejectReason (532)` field of the Order Mass Cancel Report.

Clients may use the Order Mass Cancel Request to mass cancel all orders or only those for a particular instrument, underlying or segment. Requests to mass cancel orders will be applied across all order books of the affected instruments.

A mass cancel request may apply to all the orders of the trading firm or only to those of a particular trading mnemonic. If the target party is not specified, the server will apply the request to the orders submitted by the client.

**Mass Cancel Request Types:**

| Target Party | Submitting Mnemonic | Other Mnemonic |
|--------------|-------------------|----------------|
| **All Orders** | `MassCancelRequestType (530) = 7` | `MassCancelRequestType (530) = 7`<br>`TargetPartyRole (1464) = 53`<br>`TargetPartyID (1462)` |
| **All Orders for an Instrument** | `MassCancelRequestType (530) = 1`<br>`Symbol (55)` | `MassCancelRequestType (530) = 1`<br>`Symbol (55)`<br>`TargetPartyRole (1464) = 53`<br>`TargetPartyID (1462)` |
| **All Orders for a Segment** | `MassCancelRequestType (530) = 9`<br>`MarketSegmentID (1300)` | `MassCancelRequestType (530) = 9`<br>`MarketSegmentID (1300)`<br>`TargetPartyRole (1464) = 53`<br>`TargetPartyID (1462)` |

##### 3.1.2.3 Amending an Order

The following attributes of a live order may be amended via the Order Cancel/Replace Request message:

1. Order quantity
2. Disclosed quantity
3. Price
4. Stop price
5. Time qualifier
6. Expiration time (GTT orders)
7. Expiration date (GTD orders)

The server will respond with an Execution Report or Order Cancel Reject to confirm or reject the amendment request respectively.

The client should identify the order being amended by either its `OrigClOrdID (41)` or `OrderID (37)`. If an Order Cancel/Replace Request contains values for both `OrigClOrdID (41)` and `OrderID (37)`, the server will only process the `OrderID (37)`.

If an order submitted under a different `SenderCompID (49)` is being amended, the Order Cancel/Replace Request should include its `OrderID (37)`. If the amendment is successful, the order will be treated as one submitted under the `SenderCompID (49)` that sent the Order Cancel/Replace Request.

An order will lose time priority if its order or disclosed quantity is increased or if its price is amended. A reduction in order or disclosed quantity of an order or the amendment of its time qualifier, expiration time or expiration date will not cause it to lose time priority.

> **Note:** Clients may not amend orders that are fully filled.

#### 3.1.3 Order Status

As specified in the FIX protocol, the `OrdStatus (39)` field is used to convey the current state of an order. If an order simultaneously exists in more than one order state, the value with highest precedence is reported as the `OrdStatus (39)`. The relevant order statuses are given below from the highest to lowest precedence.

| Value | Meaning |
|-------|---------|
| **E** | Pending Replace |
| **2** | Filled |
| **4** | Cancelled |
| **C** | Expired |
| **1** | Partially Filled |
| **0** | New |
| **8** | Rejected |
| **A** | Pending New |

> Please refer to Section 10.1.1 for process flow diagrams on the various statuses that may apply to an order.

#### 3.1.4 Execution Reports

The Execution Report message is used to communicate many different events to clients. The events are differentiated by the value in the `ExecType (150)` field as outlined below.

| ExecType | Usage | OrdStatus |
|----------|-------|-----------|
| **0** | **Order Accepted** - Indicates that a new order has been accepted. This message will also be sent unsolicited if an order was submitted by market operations on behalf of the client | 0 |
| **A** | **Order Pending** - Indicates that a new order has been forwarded to the risk management system for validation | A |
| **8** | **Order Rejected** - Indicates that an order has been rejected. The reason for the rejection is specified in the field `OrdRejReason (103)` | 8 |
| **F** | **Order Executed** - Indicates that an order has been partially or fully filled. The execution details (e.g. price and quantity) are specified | 1, 2 |
| **C** | **Order Expired** - Indicates that an order has expired in terms of its time qualifier or due to an execution limit | C |
| **4** | **Order Cancelled** - Indicates that an order cancel request has been accepted and successfully processed. This message will also be sent unsolicited if the order was cancelled by market operations. In such a scenario the Execution Report will include an `ExecRestatementReason (378)` of Market Option (8). It will not include an `OrigClOrdID (41)` | 4 |
| **5** | **Order Cancel/Replaced** - Indicates that an order cancel/replace request has been accepted and successfully processed | 0, 1 |
| **L** | **Triggered** - Indicates that a stop order has been activated and is available for execution | 0, 1, A |
| **D** | **Order Cancel/Replace by Market Operations** - Indicates that an order has been amended by market operations. The unsolicited message will include an `ExecRestatementReason (378)` of Market Option (8). It will not include an `OrigClOrdID (41)` | 0, 1 |
| **E** | **Order Cancel/Replace Pending** - Indicates that an order cancel/replace request has been forwarded to the risk management system for validation | E |
| **H** | **Trade Cancel** - Indicates that an execution has been cancelled. An `ExecRefID (19)` to identify the execution being cancelled will be included | 0, 1, 4, C, E |
| **G** | **Trade Correct** - Indicates that an execution has been corrected. The message will include an `ExecRefID (19)` to identify the execution being corrected and the updated execution details (e.g. price and quantity) | 1, 2, 4, C, E |

#### 3.1.5 Order and Execution Identifiers

##### 3.1.5.1 Client Order IDs

The server validates each `ClOrdID (11)` for uniqueness. Clients should comply with the FIX protocol and ensure unique ClOrdIDs across all messages (e.g. New Order – Single, Order Cancel Request, etc.) sent under a particular `SenderCompID (49)`. As the server supports GTD and GTC orders, clients should ensure that their ClOrdIDs are unique across trading days (e.g. embed the date within the ClOrdID). The Execution Report transmitted to reject an order due to a duplicate `ClOrdID (11)` will not include the fields `ExecID (17)`, `OrderID (37)`, `LeavesQty (151)` and `CumQty (14)`.

Clients must, in terms of the FIX protocol, specify the `ClOrdID (11)` when submitting an Order Cancel Request, Order Mass Cancel Request or Order Cancel/Replace Request.

##### 3.1.5.2 Order IDs

The server uses the `OrderID (37)` field of the Execution Report to affix the order identification numbers of the trading engine. Order IDs are unique across trading days.

In terms of the FIX protocol, unlike `ClOrdID (11)` which requires a chaining through cancel/replace requests and cancel requests, the `OrderID (37)` of an order will remain constant throughout its life.

Clients have the option of specifying the `OrderID (37)` when submitting an Order Cancel Request or Order Cancel/Replace Request.

##### 3.1.5.3 Execution IDs

The server uses the `ExecID (17)` field to affix a unique identifier for each Execution Report. ExecIDs are unique across trading days.

##### 3.1.5.4 Trade IDs

The server uses the `TrdMatchID (880)` field to affix a unique identifier for each trade. This identifier is referenced in the Trade Capture Reports published by the post trade system and the trade messages of the FAST market data feed. Trade IDs are unique across trading days. An Execution Report published to notify a client of a trade cancellation or correction includes the TradeID of the trade.

### 3.2 Odd Lot Order Book

The Chittagong Stock Exchange supports the trading of odd lot orders. A separate odd lot order book is available for a selected set of instruments for this purpose. This order book supports the submission of orders.

Messages (e.g. New Order – Single, etc.) intended for the odd lot order book should include an `OrderBook (30001)` of **Odd Lot (3)**.

### 3.3 Bulk Order Book

The Chittagong Stock Exchange supports the trading of bulk orders. A separate bulk order book is available for a selected set of instruments for this purpose. This order book supports the submission of orders.

Messages (e.g. New Order – Single, etc.) intended for the bulk order book should include an `OrderBook (30001)` of **Bulk (4)**.

Each order for the bulk order book is subject to a series of validations (e.g. greater than a minimum size, greater than a minimum value, etc.). The Chittagong Stock Exchange will retain an accepted order and match it against a contra side order for the identical instrument, price and quantity.

### 3.4 Foreign Order Book

The Chittagong Stock Exchange supports the trading of foreign orders. A separate foreign order book is available for a selected set of instruments for this purpose. This order book supports the submission of orders.

Messages (e.g. New Order – Single, etc.) intended for the foreign order book should include an `OrderBook (30001)` of **Foreign (100)**.

A foreign order will be identified as an order submitted with a client code and a valid custodian. Orders submitted with an invalid custodian will be rejected. Orders submitted to the foreign order book without a custodian and with or without a client code will be considered as local orders. A custodian may be specified via a `PartyRole (452)` of Custodian (28) and a client code may be specified via the `Account (1)` field.

### 3.5 Buy back and Issuing Auction Order Book

The Chittagong Stock Exchange supports the trading of auction orders. A separate buy back and issuing auction order book is available for a selected set of instruments for this purpose. This order book supports the submission of orders.

Messages (e.g. New Order – Single, etc.) intended for the auction order book should include an `OrderBook (30001)` of **Buy Back/Issuing (101)**.

Market operations will inform users of an auction through an announcement via the Market Data Gateway. Market Operations will then submit the defaulting party details and the auction order. Participants will be required to submit orders on the contra side of the auction order.

### 3.6 Short Sales

Participants may submit short sell orders on the normal order book. Participants submitting a short sell order will be required to also submit a contract ID giving the detail of the contract between the short sale party and the firm it hopes to obtain the shares from.

A short sell order will be identified by the `Side (54)` of **Sell Short (5)**. The contract ID may be specified via the `AgreementID (914)` field.

### 3.7 Closing Price trading session

The Chittagong Stock Exchange supports the submission and trading of orders priced at the closing price. A separate session named 'Closing Price Cross' is available for a selected set of instruments for this purpose. This session supports the submission of orders priced at the closing price.

Messages (e.g. New Order – Single, etc.) intended for the Closing price trading session should include a `TradingSessionID (336)` of **Closing Price Cross (a)** and a `TimeInForce (59)` of **Day (0)**.

### 3.8 Party Identification

| ID | Description | Relevant FIX Tags |
|----|-------------|-------------------|
| **Trading Mnemonic** | Identifier of the trading mnemonic the message is submitted under. Trading privileges are assigned at the level of trading mnemonics | `PartyRole (452) = 53` and `PartyID (448)` or `SenderCompID (49)`. If a `PartyRole (452)` of Trading Mnemonic (53) is not included in a message, the server will treat the `SenderCompID (49)` as the trading mnemonic |
| **Executing Firm** | Identifier of the trading firm the interest is submitted under | `PartyRole (452) = 1`<br>`PartyID (448)` |
| **Investor Account** | Identifier of the investor account on whose behalf the interest is submitted | `Account (1)`<br>`AccountType (581)` |
| **Custodian Identifier** | Identifier of the custodian bank a foreign order is submitted under | `PartyRole (452) = 28` and `PartyID (448)` |
| **Application Identifier** | Identifier in the case of what application orders are being submitted through | `Text (58)`<br>`OrderSource (30004)` |

### 3.9 Quotation Conventions

The limit price, stop price, bid price and offer price specified with an order will be interpreted by the server in terms of the applicable quotation convention for the instrument.

The values specified in these fields will be interpreted as the price per share for equity instruments.

### 3.10 Market Operations

#### 3.10.1 Interest Submission and Management

Market operations are able to submit an order, order cancel request, order cancel/replace request on behalf of a client.

The client will be notified of the order, cancel request or cancel/replace request submitted on its behalf if and when it is accepted. The client will not be notified if the action is rejected or queued.

> **Note:** This feature is intended to help a client manage an emergency situation and should not be relied upon as a normal business practice.

#### 3.10.2 Trade Cancellations and Corrections

The Chittagong Stock Exchange may cancel or correct any trade. The server will transmit Execution Reports to the relevant clients to notify them of a trade cancellation or correction. The trade being cancelled or corrected will be identified via the `ExecRefID (19)` field. This field will contain the `ExecID (17)` of the Execution Report that was originally transmitted to notify the client of the trade.

If an execution received by an order is cancelled or corrected to reduce the executed quantity, the cancelled/reduced quantity will either be cancelled or reinstated in the order book. If the quantity is cancelled, the order will be restated to reduce its order quantity by the cancelled/reduced quantity. The client will receive two notifications in such a scenario: one for the trade cancel/correction and another for the restatement. The `LeavesQty (151)` and `CumQty (14)` of a live order will always add up to its `OrderQty (38)`.

Market operations may also correct the price of an execution. A trade will not be corrected to increase the executed quantity.

### 3.11 Timestamps and Dates

The timestamps `SendingTime (52)`, `OrigSendingTime (122)` and `TransactTime (60)` should be in UTC and in the **YYYYMMDD-HH:MM:SS.sss** format. `ExpireTime (126)` should be in UTC and in the **YYYYMMDD-HH:MM:SS** format.

All dates (i.e. `ExpireDate (432)`, etc.) should be in the **YYYYMMDD** format and specified in the local date for the server (i.e. not in UTC).

---

## 4. Connectivity

### 4.1 CompIDs

The CompID of each client must be registered with The Chittagong Stock Exchange before FIX communications can begin. A single client may have multiple connections to the server (i.e. multiple FIX sessions, each with its own CompID).

The CompID of the server will be `<insert CompID of market>`. The messages sent to the server should contain the CompID assigned to the client in the field `SenderCompID (49)` and `<insert CompID of market>` in the field `TargetCompID (56)`. The messages sent from the server to the client will contain `<insert CompID of market>` in the field `SenderCompID (49)` and the CompID assigned to the client in the field `TargetCompID (56)`.

#### 4.1.1 Passwords

Each new CompID will be assigned a password on registration. Clients are strongly encouraged to change the password to one of their choosing via the Logon message. The acceptance of a login request indicates that the new password has been accepted. The new password will, if accepted, be effective for subsequent logins.

In terms of the password policy of The Chittagong Stock Exchange, the password of each CompID should be changed at least every **30** days. If not, the password will expire and the client will be unable to login to the server. In such a case, the client should contact The Chittagong Stock Exchange to have its password reset. The `SessionStatus (1409)` of the server's Logon message will be **Password Due to Expire (2)** for the last **5** days of a password's validity period.

### 4.2 Production IP Addresses and Ports

The IP address of each client must be registered with The Chittagong Stock Exchange before FIX communications can begin. The IP addresses and ports of the production servers are given below.

| Server | Primary |  | Backup |  |
|--------|---------|---|--------|---|
|        | **IP Address** | **Port** | **IP Address** | **Port** |
| 1 | xxx.xxx.xx.xx | xxxxx | xxx.xxx.xx.xx | xxxxx |
| 2 | xxx.xxx.xx.xx | xxxxx | xxx.xxx.xx.xx | xxxxx |
| 3 | xxx.xxx.xx.xx | xxxxx | xxx.xxx.xx.xx | xxxxx |
| 4 | xxx.xxx.xx.xx | xxxxx | xxx.xxx.xx.xx | xxxxx |

The Chittagong Stock Exchange will assign each registered client to one of the above primary IP addresses and ports and one of the above secondary IP addresses and ports.

### 4.3 Failover and Recovery

The system has been designed with fault tolerance and disaster recovery technology that ensures that trading should continue in the unlikely event of a process or site outage.

If the client is unexpectedly disconnected from the server, it should attempt to re-connect to primary site within a few seconds. The client should only attempt to connect to the secondary IP address and port if so requested by The Chittagong Stock Exchange.

### 4.4 Message Rate Throttling

The Chittagong Stock Exchange has implemented a scheme for throttling message traffic where each CompID is only permitted to submit up to a specified number of messages per second. The maximum rate may be negotiated with The Chittagong Stock Exchange.

Every message that exceeds the maximum rate of a CompID will be rejected via a Business Message Reject. Such a message will include a `BusinessRejectReason (380)` of **Other (0)** and an indication that the rejection was due to throttling in the `Text (58)` field.

A CompID will be disconnected by the server if its message rate exceeds its maximum rate more than **5** times in any **30** second duration. In such a case, the server will transmit a Logout message and immediately terminate the TCP/IP connection.

### 4.5 Mass Cancellation On Disconnect

At the request of the participant, the server can be configured to automatically cancel all live orders submitted under a CompID whenever it disconnects from the server.

> **Important:** This feature does not guarantee that all outstanding orders will be successfully cancelled as executions that occur very near the time of disconnect may not be reported to the client. During such a situation, the client should contact market operations to verify that all orders have been cancelled and all Execution Reports have been received.

The configuration of the mass cancellation on disconnect feature cannot be updated during a FIX session.

---

## 5. FIX Connections and Sessions

### 5.1 Establishing a FIX Connection

FIX connections and sessions between the client and server are maintained as specified in the FIXT protocol.

Each client will use the assigned IP address and port to establish a TCP/IP session with the server. The client will initiate a FIX session at the start of each trading day by sending the Logon message. The client will identify itself using the `SenderCompID (49)` field.

The server will validate the CompID, password and IP address of the client. Once the client is authenticated, the server will respond with a Logon message. The `SessionStatus (1409)` of this message will be **Session Active (0)**.

The client must wait for the server's Logon before sending additional messages. The server will break the TCP/IP connection if messages are received before the exchange of Logons.

If a logon attempt fails because of an invalid SenderCompID, TargetCompID, password or IP address, the server will break the TCP/IP connection with the client without sending a Logout or Reject. As the logon attempt failed, the server will not increment the next inbound message sequence number expected from the client.

If a logon attempt fails because of an expired password, a locked CompID or if logins are not currently permitted, the server will send a Logout message and then break the TCP/IP connection with the client. The server will increment the next inbound message sequence number expected from the client as well as its own outbound message sequence number.

### 5.2 Maintaining a FIX Session

#### 5.2.1 Message Sequence Numbers

As outlined in the FIXT protocol, the client and server will each maintain a separate and independent set of incoming and outgoing message sequence numbers. Sequence numbers should be initialized to 1 (one) at the start of the FIX session and be incremented throughout the session.

Monitoring sequence numbers will enable parties to identify and react to missed messages and to gracefully synchronize applications when reconnecting during a FIX session.

If any message sent by the client contains a sequence number that is less than what is expected and the `PossDupFlag (43)` is not set to "Y", the server will send a Logout message and terminate the FIX connection. The Logout will contain the next expected sequence number in the `Text (58)` field.

A FIX session will not continue to the next trading day. The server will initialize its sequence numbers at the start of each day. The client is expected to employ the same logic.

#### 5.2.2 Heartbeats

The client and server will use the Heartbeat message to exercise the communication line during periods of inactivity and to verify that the interfaces at each end are available. The heartbeat interval will be the `HeartBtInt (108)` specified in the client's Logon message.

The server will send a Heartbeat anytime it has not transmitted a message for the heartbeat interval. The client is expected to employ the same logic.

If the server detects inactivity for a period longer than the heartbeat interval plus a reasonable transmission time, it will send a Test Request message to force a Heartbeat from the client. If a response to the Test Request is not received by a reasonable transmission time, the server will send a Logout and break the TCP/IP connection. The client is expected to employ similar logic if inactivity is detected on the part of the server.

#### 5.2.3 Increasing Expected Sequence Number

The client or server may use the Sequence Reset message in Gap Fill mode if it wishes to increase the expected incoming sequence number of the other party.

The client or server may also use the Sequence Reset message in Sequence Reset mode if it wishes to increase the expected incoming sequence number of the other party. The `MsgSeqNum (34)` in the header of such a message will be ignored. The Sequence Reset mode should only be used to recover from an emergency situation. It should not be relied upon as a regular practice.

### 5.3 Terminating a FIX Connection

The client is expected to terminate each FIX connection at the end of each trading day before the server shuts down. The client will terminate a connection by sending the Logout message. The server will respond with a Logout to confirm the termination. The client will then break the TCP/IP connection with the server. As recommended in the FIXT protocol, clients are advised to transmit a Test Request, to force a Heartbeat from the server, before initiating the logout process.

All open TCP/IP connections will be terminated by the server when it shuts down (a Logout will not be sent). Under exceptional circumstances the server may initiate the termination of a connection during the trading day by sending the Logout message. The server will terminate the TCP/IP connection (a Logout will not be sent) if the number of messages that are buffered for a client exceeds **1,000**.

If, during the exchange of Logout messages, the client or server detects a sequence gap, it should send a Resend Request.

### 5.4 Re-Establishing a FIX Session

If a FIX connection is terminated during the trading day it may be re-established via an exchange of Logon messages. Once the FIX session is re-established, the message sequence numbers will continue from the last message successfully transmitted prior to the termination.

#### 5.4.1 Resetting Sequence Numbers: Starting a New FIX Session

##### 5.4.1.1 Reset Initiated by the Client

If the client requires both parties to initialize (i.e. reset to 1) sequence numbers, it may use the `ResetSeqNumFlag (141)` field of the Logon message. The server will respond with a Logon with the `ResetSeqNumFlag (141)` field set to "Y" to confirm the initialization of sequence numbers.

A client may also manually inform market operations that it would like the server to initialize its sequence numbers prior to the client's next login attempt.

> **Note:** These features are intended to help a client manage an emergency situation. Initializing sequence numbers on a re-login should not be relied upon as a regular practice.

##### 5.4.1.2 Reset Initiated by the Server

The system has been designed with fault tolerance and disaster recovery technology that should ensure that the server retains its incoming and outgoing message sequence numbers for each client in the unlikely event of an outage.

However, clients are required to support a manual request by The Chittagong Stock Exchange to initialize sequence numbers prior to the next login attempt.

---

## 6. Recovery

### 6.1 Resend Requests

The client may use the Resend Request message to recover lost messages. As outlined in the FIXT protocol, this message may be used in one of three modes:

1. **To request a single message** - The `BeginSeqNo (7)` and `EndSeqNo (16)` should be the same
2. **To request a specific range of messages** - The `BeginSeqNo (7)` should be the first message of the range and the `EndSeqNo (16)` should be the last of the range
3. **To request all messages after a particular message** - The `BeginSeqNo (7)` should be the sequence number immediately after that of the last processed message and the `EndSeqNo (16)` should be zero (0)

The server caches the last **1,000** messages transmitted to each CompID. Clients are unable to use a Resend Request to recover messages not in the server's cache.

### 6.2 Possible Duplicates

The server handles possible duplicates according to the FIX protocol. The client and server will use the `PossDupFlag (43)` field to indicate that a message may have been previously transmitted with the same `MsgSeqNum (34)`.

### 6.3 Possible Resends

#### 6.3.1 Client-Initiated Messages

The server does not handle possible resends for client-initiated messages (e.g. New Order – Single, etc.) and ignores the value in the `PossResend (97)` field of such messages.

#### 6.3.2 Server-Initiated Messages

The server may, in the circumstances outlined in Section 6.4 use the `PossResend (97)` field to indicate that an application message may have already been sent under a different `MsgSeqNum (34)`. The client should validate the contents (e.g. ExecID) of such a message against those of messages already received during the current trading day to determine whether the new message should be ignored or processed.

### 6.4 Transmission of Missed Messages

The Execution Report, Order Cancel Reject, Order Mass Cancel Report, and Business Message Reject messages generated during a period when a client is disconnected from the server will be sent to the client when it next reconnects. In the unlikely event the disconnection was due to an outage of the server, all such messages will include a `PossResend (97)` of "Y".

---

## 7. Message Formats

This section provides details on the header and trailer, the seven administrative messages and seventeen application messages utilized by the server. Client-initiated messages not included in this section are rejected by the server via a Reject or Business Message Reject.

### 7.1 Supported Message Types

#### 7.1.1 Administrative Messages

All administrative messages may be initiated by either the client or the server.

| Message | MsgType | Usage |
|---------|---------|-------|
| **Logon** | A | Allows the client and server to establish a FIX session |
| **Logout** | 5 | Allows the client and server to terminate a FIX session |
| **Heartbeat** | 0 | Allows the client and server to exercise the communication line during periods of inactivity and verify that the interfaces at each end are available |
| **Test Request** | 1 | Allows the client or server to request a response from the other party if inactivity is detected |
| **Resend Request** | 2 | Allows for the recovery of messages lost during a malfunction of the communications layers |
| **Reject** | 3 | Used to reject a message that does not comply with FIXT |
| **Sequence Reset** | 4 | Allows the client or server to increase the expected incoming sequence number of the other party |

#### 7.1.2 Application Messages: Order Handling

##### 7.1.2.1 Client-Initiated

| Message | MsgType | Usage |
|---------|---------|-------|
| **New Order – Single** | D | Allows the client to submit a new order |
| **Order Cancel Request** | F | Allows the client to cancel a live order |
| **Order Mass Cancel Request** | q | Allows the client to mass cancel:<br>(i) All live orders<br>(ii) All live orders for a particular instrument<br>(iii) All live orders for a particular underlying<br>(iv) All live orders for a particular segment<br><br>The mass cancel may apply to the orders of a particular trading mnemonic or to all orders of the firm |
| **Order Cancel/Replace Request** | G | Allows the client to cancel/replace a live order |

##### 7.1.2.2 Server-Initiated

| Message | MsgType | Usage |
|---------|---------|-------|
| **Execution Report** | 8 | Indicates one of the following:<br>(i) Order accepted<br>(ii) Order pending<br>(iii) Order rejected<br>(iv) Order executed<br>(v) Order expired<br>(vi) Order cancelled<br>(vii) Order cancelled/replaced<br>(viii) Order cancel/replace pending<br>(ix) Trade cancelled<br>(x) Trade corrected |
| **Order Cancel Reject** | 9 | Indicates that an order cancel request or order cancel/replace request has been rejected |
| **Order Mass Cancel Report** | r | Indicates one of the following:<br>(i) Mass order cancel request accepted<br>(ii) Mass order cancel request rejected |

#### 7.1.3 Application Messages: Other

##### 7.1.3.1 Server-Initiated

| Message | MsgType | Usage |
|---------|---------|-------|
| **Business Message Reject** | j | Indicates that an application message could not be processed |

### 7.2 Variations from the FIX Protocol

The server conforms to the FIX protocol except as follows:

1. Many of the order messages include the custom field `OrderBook (30001)`. The data type of this field is Int (i.e. integer)
2. The `AccountType (581)` field of the New Order – Single and Execution Report messages includes the custom value **Custodian (100)**
3. The Order Cancel Reject and Order Mass Cancel Report message includes the field `ApplID (1180)`
4. The Order Cancel Reject message includes the `NoPartyIDs (453)` block which was introduced in Extension Pack 115
5. The Execution Report message includes the custom value `ClientText (31000)`
6. The `TradingSessionID (336)` field of the New Order – Single and Execution Report messages includes the custom value **Closing Price Cross (a)**

### 7.3 Message Header and Trailer

#### 7.3.1 Message Header

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| 8 | BeginString | Y | FIXT.1.1 |
| 9 | BodyLength | Y | Number of characters after this field up to and including the delimiter immediately preceding the CheckSum |
| 35 | MsgType | Y | Message type |
| 49 | SenderCompID | Y | CompID of the party sending the message |
| 56 | TargetCompID | Y | CompID of the party the message is sent to |
| 34 | MsgSeqNum | Y | Sequence number of the message |
| 43 | PossDupFlag | N | Whether the message was previously transmitted under the same MsgSeqNum (34). Absence of this field is interpreted as Original Transmission (N)<br><br>**Values:**<br>Y = Possible Duplicate<br>N = Original Transmission |
| 97 | PossResend | N | Whether the message was previously transmitted under a different MsgSeqNum (34). Absence of this field is interpreted as Original Transmission (N)<br><br>**Values:**<br>Y = Possible Resend<br>N = Original Transmission |
| 52 | SendingTime | Y | Time the message was transmitted |
| 122 | OrigSendingTime | N | Time the message was originally transmitted. If the original time is not available, this should be the same value as SendingTime (52). Required if PossDupFlag (43) is Possible Duplicate (Y) |
| 1128 | ApplVerID | N | Version of FIX used in the message. Required if the message is generated by the server<br><br>**Values:**<br>9 = FIX50SP2 |

#### 7.3.2 Message Trailer

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| 10 | CheckSum | Y |  |

### 7.4 Administrative Messages

#### 7.4.1 Logon

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | A = Logon |
| | **Message Body** | | |
| 98 | EncryptMethod | Y | Method of encryption<br><br>**Values:**<br>0 = None |
| 108 | HeartBtInt | Y | Indicates the heartbeat interval in seconds |
| 141 | ResetSeqNumFlag | N | Indicates whether the client and server should reset sequence numbers. Absence of this field is interpreted as Do Not Reset Sequence Numbers (N)<br><br>**Values:**<br>Y = Reset Sequence Numbers<br>N = Do Not Reset Sequence Numbers |
| 554 | Password | N | Password assigned to the CompID. Required if the message is generated by the client |
| 925 | NewPassword | N | New password for the CompID |
| 1409 | SessionStatus | N | Status of the FIX session. Required if the message is generated by the server<br><br>**Values:**<br>0 = Session Active<br>2 = Password Due to Expire |
| 1137 | DefaultApplVerID | Y | Default version of FIX messages used in this session<br><br>**Values:**<br>9 = FIX50SP2 |
| | **Standard Trailer** | | |

#### 7.4.2 Logout

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | 5 = Logout |
| | **Message Body** | | |
| 1409 | SessionStatus | N | Status of the FIX session. Required if the message is generated by the server<br><br>**Values:**<br>4 = Session logout complete<br>6 = Account locked<br>7 = Logons are not allowed at this time<br>8 = Password expired<br>100 = Other<br>101 = Logout due to session level failure<br>102 = Logout by market operations |
| 58 | Text | N | Text specifying reason for the logout |
| | **Standard Trailer** | | |

#### 7.4.3 Heartbeat

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | 0 = Heartbeat |
| | **Message Body** | | |
| 112 | TestReqID | N | Required if the heartbeat is a response to a Test Request. The value in this field should echo the TestReqID (112) received in the Test Request |
| | **Standard Trailer** | | |

#### 7.4.4 Test Request

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | 1 = Test Request |
| | **Message Body** | | |
| 112 | TestReqID | Y | Identifier for the request |
| | **Standard Trailer** | | |

#### 7.4.5 Resend Request

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | 2 = Resend Request |
| | **Message Body** | | |
| 7 | BeginSeqNo | Y | Sequence number of first message in range |
| 16 | EndSeqNo | Y | Sequence number of last message in range |
| | **Standard Trailer** | | |

#### 7.4.6 Reject

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | 3 = Reject |
| | **Message Body** | | |
| 45 | RefSeqNum | Y | MsgSeqNum (34) of the rejected message |
| 372 | RefMsgType | N | MsgType (35) of the rejected message |
| 371 | RefTagID | N | If a message is rejected due to an issue with a particular field its tag number will be indicated |
| 373 | SessionRejectReason | N | Code specifying the reason for the reject. Please refer to Section 9.2.1 for a list of reject codes |
| 58 | Text | N | Text specifying the reason for the rejection |
| | **Standard Trailer** | | |

#### 7.4.7 Sequence Reset

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | 4 = Sequence Reset |
| | **Message Body** | | |
| 36 | NewSeqNo | Y | Sequence number of the next message to be transmitted |
| 123 | GapFillFlag | N | Mode in which the message is being used. Absence of this field is interpreted as Sequence Reset (N)<br><br>**Values:**<br>Y = Gap Fill<br>N = Sequence Reset |
| | **Standard Trailer** | | |

### 7.5 Application Messages: Order Handling

#### 7.5.1 New Order – Single

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | D = New Order - Single |
| | **Message Body** | | |
| 11 | ClOrdID | Y | Client specified identifier of the order |
| 453 | NoPartyIDs | N | Number of party identifiers |
| 448 | PartyID | N | Identifier of the party |
| 447 | PartyIDSource | N | Required if PartyID (448) is specified<br><br>**Values:**<br>D = Proprietary/Custom Code |
| 452 | PartyRole | N | Role of the PartyID (448). Required if PartyID (448) is specified<br><br>**Values:**<br>53 = Trading Mnemonic<br>28 = Custodian |
| 1 | Account | N | Identifier of the investor account on whose behalf the order is submitted |
| 581 | AccountType | N | Type of the investor account<br><br>**Values:**<br>1 = Client<br>3 = House<br>100 = Custodian |
| 55 | Symbol | Y | Identifier of the instrument |
| 30001 | OrderBook | N | Identifier of the order book. Absence of this field is interpreted as Regular (1)<br><br>**Values:**<br>1 = Regular<br>3 = Odd Lot<br>4 = Bulk<br>100 = Foreign<br>101 = Buy Back/Issuing |
| 40 | OrdType | Y | Type of the order<br><br>**Values:**<br>1 = Market<br>K = Market to Limit<br>2 = Limit<br>3 = Stop<br>4 = Stop Limit |
| 59 | TimeInForce | N | Time qualifier of the order. Absence of this field is interpreted as Day (0)<br><br>**Values:**<br>0 = Day<br>1 = Good Till Cancel (GTC)<br>2 = At the Open (OPG)<br>3 = Immediate or Cancel (IOC)<br>4 = Fill or Kill (FOK)<br>6 = Good Till Date (GTD) |
| 126 | ExpireTime | N | Time the order expires which must be a time during the current trading day. Required if TimeInForce (59) is GTD (6) and ExpireDate (432) is not specified |
| 432 | ExpireDate | N | Date the order expires. Required if TimeInForce (59) is GTD (6) and ExpireTime (126) is not specified |
| 54 | Side | Y | Side of the order<br><br>**Values:**<br>1 = Buy<br>2 = Sell<br>5 = Sell Short |
| 38 | OrderQty | Y | Total order quantity |
| 1138 | DisplayQty | N | Maximum quantity that may be displayed |
| 1084 | DisplayMethod | N | Whether the order is a reserve order<br><br>**Values:**<br>4 = Undisclosed (Reserve Order) |
| 110 | MinQty | N | Minimum quantity that must be filled |
| 44 | Price | N | Limit price. Required if OrderType (40) is Limit (2) or Stop Limit (4) |
| 99 | StopPx | N | Stop price. Required if OrderType (40) is Stop (3) or Stop Limit (4) |
| 528 | OrderCapacity | N | Capacity of the order. Absence of this field is interpreted as Agency (A)<br><br>**Values:**<br>A = Agency<br>P = Principal |
| 60 | TransactTime | Y | Time the order was created |
| 914 | AgreementID | N | Required if Side (54) is Sell Short (5) |
| 30004 | OrderSource | N | Free Format String used to identify the application from which the new order was submitted from |
| 386 | NoTradingSessions | N | Number of trading sessions. This is equal to '1' in the case of orders being submitted for the closing price trading session |
| 336 | TradingSessionID | N | Identifier of the session. Required if the order is being submitted for the closing price trading session<br><br>**Values:**<br>a = Closing Price Cross |
| | **Standard Trailer** | | |

#### 7.5.2 Order Cancel Request

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | F = Order Cancel Request |
| | **Message Body** | | |
| 11 | ClOrdID | Y | Client specified identifier of the cancel request |
| 41 | OrigClOrdID | N | ClOrdID (11) of the order being cancelled. Required if OrderID (37) is not specified |
| 37 | OrderID | N | Server specified identifier of the order being cancelled. Required if OrigClOrdID (41) is not specified |
| | Component Block<br>&lt;Trading Mnemonic&gt; | N | Identifier of the trading mnemonic |
| 55 | Symbol | Y | Must match the values in the order |
| 30001 | OrderBook | N | Identifier of the order book. Absence of this field is interpreted as Regular (1)<br><br>**Values:**<br>1 = Regular<br>3 = Odd Lot<br>4 = Bulk<br>100 = Foreign<br>101 = Buy Back/Issuing |
| 54 | Side | Y | Must match the value in the order |
| 60 | TransactTime | Y | Time the order cancel request was created |
| 30004 | OrderSource | N | Free Format String used to identify the application from which the order cancellation was submitted from |
| | **Standard Trailer** | | |

#### 7.5.3 Order Mass Cancel Request

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | q = Order Mass Cancel Request |
| | **Message Body** | | |
| 11 | ClOrdID | Y | Client specified identifier of mass cancel request |
| 530 | MassCancelRequestType | Y | Scope of the mass cancel request<br><br>**Values:**<br>1 = Cancel All Orders for Instrument<br>7 = Cancel All Orders<br>9 = Cancel All Orders for Segment |
| 1461 | NoTargetPartyIDs | N | Number of parties the mass cancel relates to. If specified, the value in this field will always be "1" |
| 1462 | TargetPartyID | N | Identifier of the party the mass cancel relates to. Required if NoTargetPartyIDs (1461) is specified |
| 1463 | TargetPartyIDSource | N | Required if NoTargetPartyIDs (1461) is specified<br><br>**Values:**<br>D = Proprietary/Custom Code |
| 1464 | TargetPartyRole | N | Role of the TargetPartyID (1462). Required if NoTargetPartyIDs (1461) is specified<br><br>**Values:**<br>1 = Executing Firm<br>53 = Trading Mnemonic |
| 55 | Symbol | N | Identifier of the instrument the mass cancel relates to. Required if MassCancelRequestType (530) is Cancel All for Instrument (1) |
| 1300 | MarketSegmentID | N | Identifier of the segment the mass cancel relates to. Please refer to Section 8 for the valid segments. Required if MassCancelRequestType (530) is Cancel All for Segment (9) |
| 60 | TransactTime | Y | Time the mass cancel request was created |
| | **Standard Trailer** | | |

#### 7.5.4 Order Cancel/Replace Request

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | G = Order Cancel/Replace Request |
| | **Message Body** | | |
| 11 | ClOrdID | Y | Client specified identifier of the cancel/replace request |
| 41 | OrigClOrdID | N | ClOrdID (11) of the order being amended. Required if OrderID (37) is not specified |
| 37 | OrderID | N | Server specified identifier of the order being amended. Required if OrigClOrdID (41) is not specified |
| 453 | NoPartyIDs | N | Number of party identifiers |
| 448 | PartyID | N | Identifier of the party |
| 447 | PartyIDSource | N | Required if PartyID (448) is specified<br><br>**Values:**<br>D = Proprietary/Custom Code |
| 452 | PartyRole | N | Role of the PartyID (448). Required if PartyID (448) is specified<br><br>**Values:**<br>53 = Trading Mnemonic<br>28 = Custodian |
| 55 | Symbol | Y | Must match the values in the order |
| 30001 | OrderBook | N | Identifier of the order book. Absence of this field is interpreted as Regular (1)<br><br>**Values:**<br>1 = Regular<br>3 = Odd Lot<br>4 = Bulk<br>100 = Foreign<br>101 = Buy Back/Issuing |
| 40 | OrdType | Y | Must match the value in the order |
| 59 | TimeInForce | N | Time qualifier of the order. Absence of this field is interpreted as Day (0)<br><br>**Values:**<br>0 = Day<br>1 = Good Till Cancel (GTC)<br>2 = At the Open (OPG)<br>3 = Immediate or Cancel (IOC)<br>4 = Fill or Kill (FOK)<br>6 = Good Till Date (GTD) |
| 126 | ExpireTime | N | Time the order expires which must be a time during the current trading day. Required if TimeInForce (59) is GTD (6) and ExpireDate (432) is not specified |
| 432 | ExpireDate | N | Date the order expires. Required if TimeInForce (59) is GTD (6) and ExpireTime (126) is not specified |
| 54 | Side | Y | Must match the value in the order |
| 38 | OrderQty | Y | Total order quantity |
| 1138 | DisplayQty | N | Maximum quantity that may be displayed |
| 1084 | DisplayMethod | N | Whether the order is a reserve order<br><br>**Values:**<br>4 = Undisclosed (Reserve Order) |
| 44 | Price | N | Limit price. Required if OrderType (40) is Limit (2) or Stop Limit (4) |
| 99 | StopPx | N | Stop price. Required if OrderType (40) is Stop (3) or Stop Limit (4) |
| 60 | TransactTime | Y | Time the cancel/replace request was created |
| 914 | AgreementID | N | Required if Side (54) is Sell Short (5) |
| 30004 | OrderSource | N | Free Format String used to identify the application from which the order modification was submitted from |
| 386 | NoTradingSessions | N | Number of trading sessions. This is equal to '1' in the case of orders being submitted for the closing price trading session |
| 336 | TradingSessionID | N | Identifier of the session. Required if the order is being submitted for the closing price trading session<br><br>**Values:**<br>a = Closing Price Cross |
| | **Standard Trailer** | | |

#### 7.5.5 Execution Report

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | 8 = Execution Report |
| | **Message Body** | | |
| 1180 | ApplID | Y | Identity of the partition |
| 17 | ExecID | Y | Server specified identifier of the message |
| 11 | ClOrdID | Y | Client specified identifier of the order |
| 41 | OrigClOrdID | N | OrigClOrdID (41), if any, that was submitted with the order cancel or cancel/replace request |
| 37 | OrderID | Y | Server specified identifier of the order |
| 442 | MultiLegReportingType | N | Type of instrument the message is generated for. Absence of this field is interpreted as Single Security (1)<br><br>**Values:**<br>1 = Single Security |
| 150 | ExecType | Y | Reason the execution report was generated<br><br>**Values:**<br>0 = New<br>4 = Cancelled<br>5 = Replaced<br>8 = Rejected<br>A = Pending New<br>C = Expired<br>D = Restated<br>E = Pending Replace<br>F = Trade<br>G = Trade Correct<br>H = Trade Cancel<br>L = Triggered |
| 880 | TrdMatchID | N | Identifier of the trade. Required if ExecType (150) is Trade (F), Trade Correct (G) or Trade Cancel (H) |
| 19 | ExecRefID | N | Reference to the execution being cancelled or corrected. Required if ExecType (150) is Trade Cancel (H) or Trade Correct (G) |
| 378 | ExecRestatementReason | N | Reason the order was restated. Required if ExecType (150) is Restated (D)<br><br>**Values:**<br>8 = Market Option |
| 39 | OrdStatus | Y | Current status of the order<br><br>**Values:**<br>0 = New<br>1 = Partially Filled<br>2 = Filled<br>4 = Cancelled<br>8 = Rejected<br>A = Pending New<br>C = Expired<br>E = Pending Replace |
| 636 | WorkingIndicator | N | Whether the order is currently being worked<br><br>**Values:**<br>N = Order is Not in a Working State<br>Y = Order is Being Worked |
| 103 | OrdRejReason | N | Code specifying the reason for the reject. Please refer to Section 9.1.1 for a list of reject codes. Required if ExecType (150) is Rejected (8) |
| 58 | Text | N | Text specifying the reason for the rejection or expiration |
| 32 | LastQty | N | Quantity executed in this fill. Required if ExecType (150) is Trade (F) or Trade Correct (G) |
| 31 | LastPx | N | Price of this fill. Required if ExecType (150) is Trade (F) or Trade Correct (G) |
| 151 | LeavesQty | Y | Quantity available for further execution. Will be "0" if OrdStatus (39) is Filled (2), Cancelled (4), Rejected (8) or Expired (C) |
| 14 | CumQty | Y | Total cumulative quantity filled |
| 55 | Symbol | Y | Identifier of the instrument |
| 30001 | OrderBook | Y | Identifier of the order book<br><br>**Values:**<br>1 = Regular<br>3 = Odd Lot<br>4 = Bulk<br>100 = Foreign<br>101 = Buy Back/Issuing |
| 453 | NoPartyIDs | N | Number of party identifiers |
| 448 | PartyID | N | Identifier of the party |
| 447 | PartyIDSource | N | Required if PartyID (448) is specified<br><br>**Values:**<br>D = Proprietary/Custom Code |
| 452 | PartyRole | N | Role of the PartyID (448). Required if PartyID (448) is specified<br><br>**Values:**<br>1 = Executing Firm<br>53 = Trading Mnemonic<br>28 = Custodian |
| 1 | Account | N | Identifier of the investor account |
| 581 | AccountType | N | Type of the investor account<br><br>**Values:**<br>1 = Client<br>3 = House<br>100 = Custodian |
| 40 | OrdType | Y | Value submitted with the order |
| 59 | TimeInForce | N | Value submitted with the order |
| 126 | ExpireTime | N | Value submitted with the order |
| 432 | ExpireDate | N | Value submitted with the order |
| 54 | Side | Y | Value submitted with the order |
| 38 | OrderQty | Y | Value submitted with the order |
| 1138 | DisplayQty | N | Quantity currently displayed in the order book |
| 1084 | DisplayMethod | N | Value submitted with the order |
| 110 | MinQty | N | Value submitted with the order |
| 44 | Price | N | Value submitted with the order |
| 99 | StopPx | N | Value submitted with the order |
| 528 | OrderCapacity | Y | Capacity of the order<br><br>**Values:**<br>A = Agency<br>P = Principal |
| 60 | TransactTime | Y | Time the transaction represented by the Execution Report occurred |
| 914 | AgreementID | N | Value submitted with the order |
| 30004 | OrderSource | N | Used to display the string entered in the free text field of the New Order – Single, Order Cancel or Order Cancel/Replace request messages |
| 336 | TradingSessionID | N | Identifier of the session. Required if the execution report is generated for orders in the closing price trading session<br><br>**Values:**<br>a = Closing Price Cross |
| 236 | Yield | N | Implied yield of this fill. Required if LastPx (31) is specified and the trade is for a fixed income instrument quoted on price |
| 159 | AccruedInterestAmt | N | Accrued Interest for a unit trade if executed on the current trading day. Multiplying this by the LastPx(31) if available, gives the Total accrued interest. May only apply for a Regular Coupon Bond |
| | **Standard Trailer** | | |

#### 7.5.6 Order Cancel Reject

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | 9 = Order Cancel Reject |
| | **Message Body** | | |
| 1180 | ApplID | Y | Identity of the partition |
| 11 | ClOrdID | Y | ClOrdID (11) that was submitted with the order cancel or cancel/replace request being rejected |
| 41 | OrigClOrdID | N | OrigClOrdID (41), if any, that was submitted with the order cancel or cancel/replace request being rejected |
| 37 | OrderID | Y | Server specified identifier of the order for which the cancel or cancel/replace was submitted. Will be "NONE" if the order is unknown |
| | Component Block<br>&lt;Trading Mnemonic&gt; | N | Values specified in the order cancel or cancel/replace request |
| 39 | OrdStatus | Y | Current status of the order. Will be Rejected (8) if the order is unknown or the request cannot be processed<br><br>**Values:**<br>0 = New<br>1 = Partially Filled<br>2 = Filled<br>4 = Cancelled<br>8 = Rejected<br>A = Pending New<br>C = Expired<br>E = Pending Replace |
| 434 | CxlRejResponseTo | Y | Type of request being rejected<br><br>**Values:**<br>1 = Order Cancel Request<br>2 = Order Cancel/Replace Request |
| 102 | CxlRejReason | Y | Code specifying the reason for the rejection. Please refer to Section 9.1.2 for a list of reject codes |
| 58 | Text | N | Text specifying the reason for the rejection |
| 60 | TransactTime | Y | Time the reject was generated |
| | **Standard Trailer** | | |

#### 7.5.7 Order Mass Cancel Report

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | r = Order Mass Cancel Report |
| | **Message Body** | | |
| 1180 | ApplID | Y | Identity of the partition |
| 1369 | MassActionReportID | Y | Server specified identifier of the message |
| 11 | ClOrdID | Y | Client specified identifier of mass cancel request |
| 530 | MassCancelRequestType | Y | Value specified in the mass cancel request |
| 531 | MassCancelResponse | Y | Action taken by server<br><br>**Values:**<br>0 = Mass Cancel Request Rejected<br>1 = Cancelled All Orders for Instrument<br>7 = Cancelled All Orders<br>9 = Cancelled All Orders for Segment |
| 532 | MassCancelRejectReason | N | Code specifying the reason for the rejection. Please refer to Section 9.1.3 for a list of reject codes. Required if MassCancelResponse (531) is Mass Cancel Request Rejected (0) |
| | **Standard Trailer** | | |

### 7.6 Application Messages: Others

#### 7.6.1 Business Message Reject

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| | **Standard Header** | | |
| 35 | MsgType | Y | j = Business Message Reject |
| | **Message Body** | | |
| 379 | BusinessRejectRefID | N | Client specified identifier (e.g. ClOrdID, etc.) of the rejected message if it is available |
| 45 | RefSeqNum | Y | MsgSeqNum (34) of the rejected message |
| 372 | RefMsgType | Y | MsgType (35) of the rejected message |
| 371 | RefTagID | N | If a message is rejected due to an issue with a particular field its tag number will be indicated |
| 380 | BusinessRejectReason | Y | Code specifying the reason for the rejection. Please refer to Section 9.2.2 for a list of reject codes |
| 58 | Text | N | Text specifying the reason for the rejection |
| | **Standard Trailer** | | |

### 7.7 Components of Application Messages

#### 7.7.1 Trading Mnemonic

| Tag | Field Name | Req | Description |
|-----|------------|-----|-------------|
| 453 | NoPartyIDs | N | Number of party identifiers. If specified, the value in this field will always be "1" |
| 448 | PartyID | N | Identifier of the party. Required if NoPartyIDs (448) is specified |
| 447 | PartyIDSource | N | Required if PartyID (448) is specified<br><br>**Values:**<br>D = Proprietary/Custom Code |
| 452 | PartyRole | N | Role of the specified PartyID (448). Required if PartyID (448) is specified<br><br>**Values:**<br>53 = Trading Mnemonic |

---

## 8. Segments

| Segment | Description |
|---------|-------------|
| **A** | Main Board Category 'A' |
| **B** | Secondary Board Category 'B' |
| **G** | Default Board Category 'G' |
| **N** | Category 'N' |
| **Z** | Category 'Z' |

---

## 9. Reject Codes

### 9.1 Order Handling

#### 9.1.1 Execution Report

| OrdRejReason | Meaning |
|--------------|---------|
| **2** | Exchange closed |
| **3** | Order exceeds limit (i.e. rejected by risk system) |
| **5** | Unknown order |
| **6** | Duplicate order (i.e. duplicate ClOrdID) |
| **16** | Price exceeds current price band |
| **18** | Invalid price increment |

> Please refer to the &lt;Reject Code Specification&gt; for the list of reject codes and meanings specific to The Chittagong Stock Exchange.

#### 9.1.2 Order Cancel Reject

| CxlRejReason | Meaning |
|--------------|---------|
| **1** | Unknown order |
| **6** | Duplicate ClOrdID |
| **8** | Price exceeds current price band |
| **18** | Invalid price increment |

> Please refer to the &lt;Reject Code Specification&gt; for the list of reject codes and meanings specific to The Chittagong Stock Exchange.

#### 9.1.3 Order Mass Cancel Report

| MassCancelRejectReason | Meaning |
|------------------------|---------|
| **1** | Unknown instrument |

> Please refer to the &lt;Reject Code Specification&gt; for the list of reject codes and meanings specific to The Chittagong Stock Exchange.

### 9.2 Others

#### 9.2.1 Reject

| SessionRejectReason | Meaning |
|---------------------|---------|
| **1** | Required tag missing |
| **2** | Tag not defined for this message type |
| **4** | Tag specified without a value |
| **5** | Value is incorrect (out of range) for this tag |
| **6** | Incorrect data format for value |
| **9** | CompID problem |
| **11** | Invalid MsgType |
| **13** | Tag appears more than once |
| **14** | Tag specified out of required order |
| **15** | Repeating group fields out of order |
| **18** | Invalid or unsupported application version |
| **99** | Other |

#### 9.2.2 Business Message Reject

| BusinessRejectReason | Meaning |
|---------------------|---------|
| **0** | Other |
| **2** | Unknown instrument |
| **3** | Unsupported message type |
| **4** | Application not available |
| **5** | Conditionally required field missing |

---

## 10. Process Flows

### 10.1 Order Handling

#### 10.1.1 Order Status Changes

```
                    Order Entry
                         |
                    Forwarded to
                    Risk system
                         |
               ┌─────────┴─────────┐
               │                   │
        Rejected by           Accepted by
        risk system          risk system
               │                   │
           Rejected                 |
                              ┌────┴────┐
                              │         │
                         and added   and fully
                         to order      filled
                            book         │
                              │      Filled
                             New
                              │
                    ┌─────────┼─────────┐
                    │         │         │
               Partially   Expire     Cancel
                 filled     time      request
                    │         │         │
               Partially   Expired   Cancelled
                Filled
                    │
              ┌─────┼─────┐
              │     │     │
         Fully    Expire Cancel
         filled   time   request
              │     │     │
          Filled Expired Cancelled

Replace request     Replace accepted
forwarded to   ────▶ by risk system
risk system           │
     │                │
Pending Replace   ┌───┴───┐
     │            │       │
Replace accepted  Added to Partially
by risk system   order    filled
     │            book      │
    New             │      │
                   New   Partially
                         Filled

Key:
■ Order Status    ■ Client action    ■ System action
```

#### 10.1.1.1 Market Operations Actions

```
              New ────────────────┐
               │                  │
         ┌─────┼─────┐             │
         │     │     │             │
    Trade   Order   Order          │
 cancelation cancel- reinstate-     │
or correction ation    ment         │
         │     │      │             │
    Partially Cancelled New         │
     Filled                        │
         │                         │
    ┌────┼────┐                    │
    │    │    │                    │
Trade  Order Trade                 │
cancel- cancel- cancel-            │
ation   ation  ation              │
or corr-      or corr-            │
ection         ection             │
    │    │      │                  │
Partially Cancelled Filled        │
 Filled                           │
                                  │
             Pending New ─────────┘
                  │
            ┌─────┼─────┐
            │     │     │
        Order  Filled Trade
      cancel-        cancel-
       ation         ation
            │     │     │
      Cancelled Filled Order
                     reinstate-
                       ment

Key:
■ Order Status    ■ Service Desk action
```

---

**End of Document**

*Chittagong Stock Exchange – Trading Gateway (FIX 5.0)_v2.00*