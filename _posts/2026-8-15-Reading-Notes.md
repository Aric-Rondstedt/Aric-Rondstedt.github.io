---
layout:     post
title:      Chapter 1 Introduction to TCP/IP Networking
subtitle:   Reading Notes(CCNA 200-301 Vol.1, Chapter 1)
date:       2026-8-15
author:     世维
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - network
    - Reading Notes(CCNA 200-301 Vol.1)
---
# Chapter 1: Introduction to TCP/IP Networking

*(Synthesized from all source materials for CCNA 200-301 Vol.1, Chapter 1)*

---

## Exam Topics Covered

- **1.0 Network Fundamentals**
- **1.3 Compare physical interface and cabling types**
  - **1.3.a** Single-mode fiber, multimode fiber, copper
  - **1.3.b** Connections (Ethernet shared media and point-to-point)

> **Quiz Reference**: See pages 4–6 of the original textbook for the "Do I Know This Already?" quiz (specific example questions are on page 5).

---

## Perspectives on Networking

- **Core Function**: A network's primary job is moving data from one device to another.
- **End-User View**: Often perceived as high-speed Internet access, typically via a cable modem using coaxial or fiber Ethernet, or a wireless LAN.
- **Enterprise Network**: A network built by a company or organization for employee communication and internal services.
- **SOHO (Small Office/Home Office)**: Smaller home-based or small-business networks used for commercial purposes.
- **Cloud Symbol**: In network diagrams, a cloud represents a network portion whose internal details are not relevant to the current discussion.

---

## History Leading to TCP/IP

- **Networking Model (Architecture)**: A comprehensive set of documents (protocols + physical specifications) defining how a network works—similar to an architectural blueprint.
- **Protocol Definition**: A set of logical rules that devices follow to communicate with each other.
- **Vendor-Proprietary Era (1970s–80s)**: Early models like IBM's SNA (1974) led to fragmented, isolated, and complex multi-vendor networks.
- **The OSI Model**: The International Organization for Standardization (ISO) developed the OSI model in the late 1970s as an open, vendor-neutral alternative.
- **The TCP/IP Rival**: A separate effort, stemming from a U.S. Department of Defense (DoD) contract and developed with university researchers, produced TCP/IP.
- **The Market Winner (1990s–2000s)**:
  - During the 1990s, networks used both OSI and TCP/IP.
  - By the late 1990s and into the 2000s, TCP/IP became the dominant, near-universal model.
  - **Key Success Factor**: TCP/IP succeeded largely due to a **"code-first, standardize-second"** approach, whereas OSI followed a **"standard-first, code-second"** approach that delayed deployment.
- **Modern Status**: TCP/IP is supported on virtually all operating systems and Cisco products. The current CCNA exam topics no longer explicitly reference the OSI/TCP/IP models by name, but the terminology remains essential for everyday networking discussions.

---

## Overview of the TCP/IP Networking Model

- **RFCs (Requests for Comments)**: TCP/IP protocols are defined in these publicly available documents.
- **External Standards**: TCP/IP reuses external standards where possible (e.g., IEEE defines Ethernet and 802.11 Wi-Fi; TCP/IP references them rather than redefining them).
- **Layered Architecture**: The model organizes functions into layers, each covering a category of related protocols and standards.

### The 5-Layer TCP/IP Model (Top to Bottom)

1.  **Application**
2.  **Transport**
3.  **Network** (also called the Internet layer)
4.  **Data Link**
5.  **Physical**

> **Note**: The older 4-layer TCP/IP model (RFC 1122) combines the Data Link and Physical layers into a single layer called the **Link** layer.

### Layer Function Summary

- **Physical Layer**: Defines cabling, signaling (electrical/optical), and the transmission of raw bits over a link.
- **Data-Link Layer**: Defines the rules and conventions for using the physical medium (e.g., framing, MAC addressing); provides a service to the Network layer above it. Includes Ethernet variants and 802.11 (Wi-Fi) protocols.
- **Network Layer (Internet Layer)**: Handles end-to-end delivery of data across the entire path (analogous to a postal system).
- **Transport & Application Layers**: Focus on application needs—identifying data, requesting transmission, and recovering lost data.

### Example Protocols by Layer

| TCP/IP Layer         | Example Protocols        |
| :------------------- | :----------------------- |
| Application          | HTTP, HTTPS, POP3, SMTP  |
| Transport            | TCP, UDP                 |
| Network (Internet)   | IP, ICMP                 |
| Data Link & Physical | Ethernet, 802.11 (Wi-Fi) |

---

## TCP/IP Application Layer

- **Role**: Provides services and an interface between application software and the network. It does **not** define the application itself (e.g., it defines HTTP, not the web browser).
- **HTTP (Hypertext Transfer Protocol)** :
  - Created by Tim Berners-Lee; defines how web browsers request and web servers return page content.
  - URLs and URIs often begin with "http" to indicate the use of this protocol.
  - **Basic HTTP Flow**:
    1.  Browser sends a **GET** request (HTTP header) to the server.
    2.  Server replies with an HTTP header (e.g., return code **200 = OK**) followed by the requested data.
    3.  Subsequent data-only messages may follow without repeating the header (the header is omitted for efficiency).
  - **Common Error Code**: **404 = Not Found**.

---

## TCP/IP Transport Layer

- **Main Protocols**: **TCP (Transmission Control Protocol)** and **UDP (User Datagram Protocol)**.
- **Key Service - Error Recovery (TCP)** :
  - TCP provides error recovery via **sequence numbers** and **acknowledgments**.
  - The receiver detects gaps in sequence numbers and requests retransmission of missing data.
- **Same-Layer Interaction**: Two computers' instances of the same layer communicate using protocol-defined headers to coordinate actions (e.g., sequence numbers between two TCP processes).
- **Adjacent-Layer Interaction**: A lower layer (e.g., Transport) provides a specific service (e.g., error recovery) to the layer above it (e.g., Application) on the **same** computer.

---

## TCP/IP Network Layer

- **Primary Protocol**: **IP (Internet Protocol)** — the "IP" in "TCP/IP".
- **Core Functions**: Addressing and routing.
- **Postal System Analogy**:
  - **IP Address** ≈ Unique postal address.
  - **Address Grouping** (like ZIP codes) simplifies routing decisions.
  - **Routers** ≈ Post offices that forward mail.
- **IP Address Format**: Written in **dotted-decimal notation (DDN)** — four numbers separated by periods (e.g., `1.1.1.1`).
- **IP Host**: Any device with an IP address connected to a TCP/IP network.
- **Basic IP Routing Process**:
  1.  The sending host forwards the IP packet to a local nearby router.
  2.  Each router examines the packet's destination IP address and compares it to its known routes (routing table).
  3.  The router forwards the packet toward the destination.
  4.  This process repeats hop-by-hop until the packet reaches the destination host.

---

## TCP/IP Data-Link and Physical Layers

- **Physical Layer**: Defines the physical cabling (copper, single-mode fiber, multimode fiber), connectors, and the electrical/optical signals used to transmit bits.
- **Data-Link Layer**: Defines the rules for using that physical medium (e.g., media access control) and provides a service to the Network layer.

### Four-Step Link-Layer Delivery Process (Sender to Receiver on the same link)

1.  **Encapsulation**: Sender encapsulates the IP packet inside a data-link header and trailer, creating a **frame** (e.g., Ethernet frame).
2.  **Transmission**: Sender transmits the frame's bits as physical signals (electrical or optical).
3.  **Reception**: Receiver physically receives the signal and reconstructs the bits.
4.  **De-encapsulation**: Receiver removes the data-link header and trailer to recover the original IP packet.

> **Note**: Headers appear at the beginning of a message; trailers appear at the end.

---

## Data Encapsulation Terminology

- **Encapsulation**: The process of wrapping data with headers (and sometimes trailers) at each layer as it moves down the protocol stack.

### Five-Step Encapsulation Process on the Sending Host (5-Layer Model)

1.  **Application Layer**: Create data with any needed application header.
2.  **Transport Layer**: Encapsulate the data inside a TCP or UDP header (creating a segment).
3.  **Network Layer**: Encapsulate the segment inside an IP header (creating a packet).
4.  **Data-Link Layer**: Encapsulate the packet inside a header and trailer (creating a frame).
5.  **Physical Layer**: Transmit the raw bits over the physical medium.

### Names of TCP/IP Messages (PDUs)

- **Segment** — Transport Layer PDU (TCP/UDP header + data).
- **Packet** — Network Layer PDU (IP header + data).
- **Frame** — Data-Link Layer PDU (Link Header + data + Link Trailer).
- **PDU (Protocol Data Unit)** : A generic term for any layer's message (segment, packet, and frame are all PDUs).
- When discussing a specific layer, everything encapsulated beyond that layer's own header is generically referred to as **"data."**

---

## OSI Networking Model and Terminology

- **Outcome**: OSI ultimately did not become the deployed networking model; TCP/IP won. However, **OSI terminology persists** in industry usage.
- **The 7-Layer OSI Model** (Top to Bottom):
  1.  Application
  2.  Presentation
  3.  Session
  4.  Transport
  5.  Network
  6.  Data Link
  7.  Physical

### Mapping OSI to the 5-Layer TCP/IP Model

| OSI Layer (7-Layer)                | TCP/IP Layer (5-Layer) | Common Layer Numbering |
| :--------------------------------- | :--------------------- | :--------------------- |
| Application, Presentation, Session | Application            | Layers 5–7             |
| Transport                          | Transport              | Layer 4                |
| Network                            | Network / Internet     | Layer 3                |
| Data Link                          | Data Link              | Layer 2                |
| Physical                           | Physical               | Layer 1                |

- **Industry Convention**: Even when discussing TCP/IP, networking professionals frequently reference OSI layer numbers (e.g., "Layer 7 protocol" for application-level, or "Layer 4 switch" for Transport layer operations).

---

## Key Diagrams for Review (Refer to Textbook)

- **Figure 1-10**: Basic IP routing concept.
- **Figure 1-11**: Data-link services for delivering IP packets.
- **Figure 1-12**: Five steps of data encapsulation.
- **Figure 1-13**: Meanings of segment, packet, and frame.
- **Figure 1-14**: OSI vs. TCP/IP models comparison.

---

## 专业术语中文对照表 (Glossary of Key Terms)

| English Term                    | 中文翻译                |
| :------------------------------ | :---------------------- |
| Adjacent-layer interaction      | 相邻层交互              |
| Application Layer               | 应用层                  |
| Cloud (in diagrams)             | 云（表示未知网络部分）  |
| Data Link Layer                 | 数据链路层              |
| De-encapsulation                | 解封装                  |
| Dotted-decimal notation (DDN)   | 点分十进制记法          |
| Encapsulation                   | 封装                    |
| Enterprise network              | 企业网络                |
| Frame                           | 帧                      |
| IP Host                         | IP主机                  |
| Network Layer / Internet Layer  | 网络层 / 互联网层       |
| Networking model                | 网络模型                |
| Packet                          | 数据包 / 报文           |
| PDU (Protocol Data Unit)        | 协议数据单元            |
| Physical Layer                  | 物理层                  |
| Router                          | 路由器                  |
| Routing                         | 路由                    |
| Same-layer interaction          | 同层交互                |
| Segment                         | 报文段 / 数据段         |
| SOHO (Small Office/Home Office) | 小型办公室/家庭办公室   |
| TCP/IP                          | 传输控制协议/互联网协议 |
| Transport Layer                 | 传输层                  |
| Trailer                         | 尾部 / 报尾             |
