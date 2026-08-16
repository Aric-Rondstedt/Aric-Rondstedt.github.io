---
layout:     post
title:      Chapter 2 Comprehensive Guide to Fundamentals of Ethernet LANs
subtitle:   Reading Notes(CCNA 200-301 Vol.1, Chapter 2)
date:       2026-8-16
author:     世维
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - network
    - Reading Notes(CCNA 200-301 Vol.1)
---
# Chapter 2:Comprehensive Guide to Fundamentals of Ethernet LANs

*(Synthesized from all source materials for CCNA 200-301 Vol.1, Chapter 2)*

---

## 1. Overview of LANs

### 1.1 Typical SOHO (Small Office/Home Office) LANs

- **Core device**: An Ethernet LAN switch provides physical ports for cable connections.
- **Components**: Ethernet switch + Ethernet cables + end devices (PCs, printers) + a router (connects the LAN to the WAN/Internet).
- **Consumer-grade integration**: Many SOHO devices integrate the switch, router, and wireless Access Point (AP) into a single physical device, commonly labeled as a "wireless router."
- **Wireless LAN**: Uses radio waves (IEEE 802.11 standard). A wireless AP connects wireless nodes to the wired Ethernet network via a single Ethernet link.
- **AP as a separate device**: If the AP is a standalone unit, it connects to the switch using one Ethernet cable.

### 1.2 Typical Enterprise LANs

- **Per-floor deployment**: LAN switches are installed in wiring closets on each floor. Ethernet cabling runs from the closet to cubicles and conference rooms.
- **Wireless**: Wireless APs are deployed per floor to support roaming and devices without Ethernet interfaces.
- **Inter-floor communication**: Per-floor switches connect to a centralized **distribution switch (SWD)** to enable communication between floors.
- **WAN/Internet connectivity**: A router connects the enterprise LAN to the WAN or the Internet using an Ethernet interface and cable.
- **Layer 2 vs. Layer 1**:
  - A **switch** is a Layer 2 device (forwards based on the data-link header / MAC address).
  - A **hub** is a Layer 1 device (simply repeats the electrical signal out all other ports, with no concept of frames or addresses).

---

## 2. Ethernet Physical Layer Standards

- Ethernet is a family of IEEE 802.3 standards defining both physical-layer and data-link-layer specifications.
- Speeds range from 10 Mbps to 400 Gbps.
- **Naming conventions**:
  - Informal name (e.g., Fast Ethernet).
  - Formal short name (e.g., 100BASE-T).
  - IEEE standard number (e.g., 802.3u).
- **Suffix meaning**:
  - **T** = UTP copper cabling.
  - **X** = Fiber-optic cabling.

### Common Ethernet Types (Table 2-2)

| Speed     | Common Name      | Short Standard Name | IEEE Standard | Cable Type & Max Length |
| :-------- | :--------------- | :------------------ | :------------ | :---------------------- |
| 10 Mbps   | Ethernet         | 10BASE-T            | 802.3         | Copper UTP, 100 m       |
| 100 Mbps  | Fast Ethernet    | 100BASE-T           | 802.3u        | Copper UTP, 100 m       |
| 1000 Mbps | Gigabit Ethernet | 1000BASE-LX         | 802.3z        | Fiber, 5000 m           |
| 1000 Mbps | Gigabit Ethernet | 1000BASE-T          | 802.3ab       | Copper UTP, 100 m       |
| 10 Gbps   | 10 Gig Ethernet  | 10GBASE-T           | 802.3an       | Copper UTP, 100 m       |

- **Consistent Data-Link Layer**: Regardless of the physical medium (UTP/fiber) or speed, all Ethernet standards share the **same data-link-layer frame format**. This allows a single frame to be forwarded seamlessly across mixed media types without changing the frame structure.

---

## 3. Building Physical Ethernet LANs with UTP

### 3.1 Transmitting Data Using Twisted Pairs

- Data transmission relies on electrical circuits formed by pairs of copper wires (one circuit = one twisted pair).
- **Encoding scheme**: The transmitter varies the electrical signal over time; the receiver interprets these voltage changes as binary 0s and 1s.
- **EMI cancellation**: Wires are twisted together to cancel Electromagnetic Interference (EMI) and crosstalk between pairs.

### 3.2 Components of a UTP Ethernet Link

- **Physical components**: UTP cable (color-coded twisted pairs) + RJ-45 connectors (8 pins) + RJ-45 ports on devices.
- **Pair requirements**:
  - 10BASE-T / 100BASE-T: 2 pairs.
  - 1000BASE-T: 4 pairs.
- **Modular transceivers** (used on switches for interface flexibility):
  - **GBIC (Gigabit Interface Converter)**: Original, larger Gigabit transceiver form factor.
  - **SFP (Small Form-Factor Pluggable)**: Smaller replacement for GBIC, used for Gigabit interfaces.
  - **SFP+ (Small Form-Factor Pluggable Plus)**: Same size as SFP, but used for 10-Gbps interfaces.
- *Note*: Cisco switches prominently support these modular transceiver slots for flexible port configuration.

### 3.3 UTP Cabling Pinouts for 10BASE-T and 100BASE-T

- Uses two pairs: pins 1,2 and pins 3,6.
- **Transmit/Receive pin assignments (Table 2-3)**:

| Transmits on Pins 1,2             | Transmits on Pins 3,6 |
| :-------------------------------- | :-------------------- |
| PC NICs                           | Hubs                  |
| Routers                           | Switches              |
| Wireless APs (Ethernet interface) | —                     |

- **Straight-through cable**: Pin 1→1, 2→2, 3→3, 6→6. Used when the two end devices transmit on **different** pin pairs (e.g., PC to Switch).
- **Crossover cable**: Pins 1,2 connect to 3,6 on the opposite end (and vice versa). Used when the two end devices transmit on the **same** pin pair (e.g., Switch to Switch, PC to Router).
- **Typical usage**: PC↔Switch = straight-through; Switch↔Switch = crossover.

### 3.4 Automatic Rewiring with Auto-MDIX

- Introduced with Gigabit Ethernet (1998). Auto-MDIX automatically detects an incorrect cable pinout and internally swaps the transmit/receive pairs so the link works regardless of whether a straight-through or crossover cable is used.
- This allows network plants to use all straight-through cables while the switch compensates.

### 3.5 UTP Cabling Pinouts for 1000BASE-T (Gigabit)

- Requires **4 pairs** (adding pins 4,5 and 7,8), enabling simultaneous bidirectional transmission on each pair.
- **Straight-through cable**: Maps pins 1:1 across all four pairs (1-2, 3-6, 4-5, 7-8).
- **Crossover cable**: Crosses pair A/B (1-2 ↔ 3-6) AND pair C/D (4-5 ↔ 7-8).
- The same device-grouping logic (Table 2-3) applies for determining when a crossover is needed.

---

## 4. Building Physical Ethernet LANs with Fiber

### 4.1 Fiber Transmission Concepts

- Fiber-optic cables use a fiberglass core to transmit light pulses (not electricity).
- **Cable structure** (inner to outer): Core → Cladding → Buffer → Strengthener → Outer Jacket.
- **Cladding**: Reflects light back into the core (total internal reflection) to prevent signal loss.
- **Multimode Fiber (MM)**:
  - Larger core.
  - Allows multiple angles ("modes") of light.
  - Uses LED (or cheaper laser) transmitters.
  - Shorter distances, lower cost.
- **Single-Mode Fiber (SM)**:
  - Much smaller core (~1/5 the diameter of MM).
  - Single angle of light.
  - Uses laser transmitters.
  - Supports distances up to tens of kilometers, higher cost.
- **Directionality**: A full-duplex optical link requires **two separate fiber strands**—one for transmit (Tx) and one for receive (Rx).

### 4.2 Using Fiber with Ethernet

- Requires switches with built-in optical ports or modular SFP/SFP+ slots.

**Sample 10-Gbps Fiber Standards (Table 2-4)**

| Standard    | Cable Type  | Max Distance |
| :---------- | :---------- | :----------- |
| 10GBASE-S   | Multimode   | 400 m        |
| 10GBASE-LX4 | Multimode   | 300 m        |
| 10GBASE-LR  | Single-Mode | 10 km        |
| 10GBASE-E   | Single-Mode | 30 km        |

### 4.3 UTP vs. Multimode vs. Single-Mode Comparison (Table 2-5)

| Criteria                                  | UTP   | Multimode | Single-Mode |
| :---------------------------------------- | :---- | :-------- | :---------- |
| Relative Cabling Cost                     | Low   | Medium    | Medium      |
| Relative Switch Port Cost                 | Low   | Medium    | High        |
| Approx. Max Distance                      | 100 m | 500 m     | 40 km       |
| Susceptibility to EMI                     | Some  | None      | None        |
| Risk of Signal Interception/Eavesdropping | Some  | None      | None        |

- **Tradeoffs**: UTP is cheapest but vulnerable to EMI in noisy environments (e.g., factories) and emits faint signals that pose security risks. Fiber offers superior distance, EMI immunity, and security but at a higher cost.

---

## 5. Sending Data in Ethernet Networks

### 5.1 Ethernet Frame Format

- Standardized IEEE 802.3 frame structure (Figure 2-18, Table 2-6):

| Field                       | Bytes   | Description                                               |
| :-------------------------- | :------ | :-------------------------------------------------------- |
| Preamble                    | 7       | Synchronization pattern                                   |
| SFD (Start Frame Delimiter) | 1       | Marks the start of the Destination MAC field              |
| Destination MAC Address     | 6       | Identifies the intended recipient                         |
| Source MAC Address          | 6       | Identifies the sender                                     |
| Type / EtherType            | 2       | Identifies the Layer 3 protocol inside (e.g., IPv4, IPv6) |
| Data and Pad                | 46–1500 | Encapsulated upper-layer PDU; padded to min 46 bytes      |
| FCS (Frame Check Sequence)  | 4       | Error detection (trailer field)                           |

- **Maximum Transmission Unit (MTU)**: The max IP MTU over Ethernet is **1500 bytes** (i.e., the max data field size).

### 5.2 Ethernet (MAC) Addressing

- **Length**: 6 bytes (48 bits), displayed as 12 hexadecimal digits (Cisco style: `0000.0C12.3456`).
- **Structure**:
  - **First 3 bytes**: OUI (Organizationally Unique Identifier) – assigned by IEEE to the manufacturer.
  - **Last 3 bytes**: Vendor-assigned unique value → ensures a globally unique address (BIA – Burned-In Address).
- **Alternative names**: LAN address, Hardware address, Physical address, Universal/Global address.
- **Address types**:
  - **Unicast**: Identifies a single interface.
  - **Broadcast**: `FFFF.FFFF.FFFF` – delivered to all devices on the LAN.
  - **Multicast**: Delivered to a specific subset of devices that have joined the group.

### 5.3 Identifying Network Layer Protocols with the Type Field

- The Type (EtherType) field identifies the Layer 3 protocol encapsulated in the frame.
- IEEE-managed assigned values:
  - IPv4 = `0x0800`
  - IPv6 = `0x86DD`

### 5.4 Error Detection with FCS

- **Sender**: Computes a mathematical checksum over the frame contents and stores it in the FCS trailer.
- **Receiver**: Recalculates the checksum and compares it to the FCS value.
- **Match**: No error.
- **Mismatch**: Frame is silently discarded.
- **Important**: Ethernet performs error **detection** only. Error **recovery** (e.g., retransmission) is handled by higher-layer protocols (e.g., TCP).

---

## 6. Forwarding Ethernet Frames: Switches vs. Hubs

### 6.1 Full-Duplex Logic (Modern Ethernet)

- Used exclusively in switch-based (point-to-point) networks.
- Devices can **send and receive simultaneously** (no waiting).
- No collisions occur because each link is independent.
- **Used on**: PC↔Switch and Switch↔Switch links.

### 6.2 Half-Duplex Logic and Hubs

- **Hub behavior**: A Layer 1 hub repeats an incoming electrical signal out all other ports, creating a single **shared collision domain**.
- **Collisions**: If two hub-connected devices transmit simultaneously, the signals garble and a collision occurs.
- **Requirement**: Devices attached to a hub must use **half-duplex** (cannot send while receiving).
- **CSMA/CD (Carrier Sense Multiple Access with Collision Detection) Algorithm** (used in half-duplex/hub environments):
  1. **Carrier Sense**: Listen until the media is idle.
  2. **Transmit**: Send the frame.
  3. **Detect**: Continue listening while sending to detect collisions.
  4. **On Collision**:
     - Send a jamming signal to notify all devices.
     - Each sender waits a random backoff time.
     - Retry from step 1.
- **Rule of thumb**: Switch-to-switch / switch-to-PC links = **Full duplex**. Any link touching a hub = **Half duplex**.

### 6.3 Exam Terminology Mapping (Topic 1.3.b)

- **Ethernet Shared Media**: Hub-based design. Requires CSMA/CD and half-duplex. Bandwidth is shared among all connected devices.
- **Ethernet Point-to-Point**: Switch-based design. Each link operates independently. Full-duplex allows every link to send simultaneously without collisions.

---

## 7. Hardware Transceivers (Consolidated Details)

- **GBIC (Gigabit Interface Converter)**: The original, larger form-factor transceiver for Gigabit Ethernet.
- **SFP (Small Form-Factor Pluggable)**: The smaller, modern replacement for GBIC; used on Gigabit Ethernet interfaces.
- **SFP+ (Small Form-Factor Pluggable Plus)**: Identical physical size to SFP, but designed for **10-Gbps** Ethernet interfaces.
- **Purpose**: Allow administrators to swap physical interfaces (e.g., copper vs. various fiber types) without replacing the entire switch module.

---

## 8. Exam Preparation Aid

### “Do I Know This Already?” Quiz Mapping

Based on the 9-question chapter-opening quiz:

- **Q1–2**: An Overview of LANs
- **Q3–4**: Building Physical Ethernet LANs with UTP
- **Q5**: Building Physical Ethernet LANs with Fiber
- **Q6–9**: Sending Data in Ethernet Networks

### Key Reference Figures & Tables (Page References)

| Item                 | Description                                          |
| :------------------- | :--------------------------------------------------- |
| Fig 2-3              | Enterprise wired/wireless LAN diagram                |
| Table 2-2            | Ethernet standards classification                    |
| Fig 2-9 / 2-10       | 10/100BASE-T straight-through pinout                 |
| Fig 2-11 / Table 2-3 | Crossover pinout & pin-pair grouping                 |
| Fig 2-12             | Typical use of straight-through vs. crossover cables |
| Fig 2-15             | Multimode fiber transmission concept                 |
| Table 2-5            | UTP / Multimode / Single-Mode comparison             |
| Fig 2-19             | MAC address structure (OUI + vendor)                 |
| Fig 2-21/2-23        | Full-duplex vs. Half-duplex application examples     |

---

## Appendix: Professional Terminology (Chinese Translation)

| English Term                             | Chinese Translation               |
| :--------------------------------------- | :-------------------------------- |
| 10BASE-T / 100BASE-T / 1000BASE-T        | 10BASE-T / 100BASE-T / 1000BASE-T |
| Access Point (AP)                        | 接入点                            |
| Auto-MDIX                                | 自动介质相关接口交叉              |
| Broadcast Address                        | 广播地址                          |
| Burned-In Address (BIA)                  | 烧录地址                          |
| Cladding                                 | 包层                              |
| Collision                                | 冲突                              |
| Collision Domain                         | 冲突域                            |
| Core                                     | 纤芯                              |
| Crossover Cable                          | 交叉线                            |
| CSMA/CD                                  | 载波监听多路访问/冲突检测         |
| Distribution Switch (SWD)                | 分布交换机                        |
| Electromagnetic Interference (EMI)       | 电磁干扰                          |
| Encoding Scheme                          | 编码方案                          |
| Ethernet Frame                           | 以太网帧                          |
| Fast Ethernet                            | 快速以太网                        |
| Fiber-Optic Cable                        | 光纤线缆                          |
| Frame Check Sequence (FCS)               | 帧校验序列                        |
| Full Duplex                              | 全双工                            |
| Gigabit Ethernet                         | 千兆以太网                        |
| Half Duplex                              | 半双工                            |
| Hub                                      | 集线器                            |
| IEEE 802.3                               | IEEE 802.3标准                    |
| Jamming Signal                           | 阻塞信号                          |
| Layer 1 / Layer 2 Switch                 | 一层/二层交换机                   |
| MAC Address                              | MAC地址                           |
| Maximum Transmission Unit (MTU)          | 最大传输单元                      |
| Multimode Fiber (MM)                     | 多模光纤                          |
| Network Interface Card (NIC)             | 网卡                              |
| Organizationally Unique Identifier (OUI) | 组织唯一标识符                    |
| Point-to-Point                           | 点对点                            |
| Preamble                                 | 前导码                            |
| RJ-45 Connector                          | RJ-45连接器                       |
| Router                                   | 路由器                            |
| Shared Media                             | 共享介质                          |
| Single-Mode Fiber (SM)                   | 单模光纤                          |
| Start Frame Delimiter (SFD)              | 帧起始定界符                      |
| Straight-Through Cable                   | 直通线                            |
| Transceiver                              | 收发器                            |
| Type Field / EtherType                   | 类型字段                          |
| Unicast Address                          | 单播地址                          |
| Unshielded Twisted-Pair (UTP)            | 非屏蔽双绞线                      |
| Wired LAN                                | 有线局域网                        |
| Wireless LAN                             | 无线局域网                        |
