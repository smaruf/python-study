"""Enhanced FIX protocol server for Phase 2."""
import asyncio
import socket
from typing import Dict, Optional
from datetime import datetime


class FIXSession:
    """FIX session management."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_active = False
        self.last_heartbeat = datetime.utcnow()
        self.seq_num_in = 1
        self.seq_num_out = 1


class FIXServer:
    """Enhanced FIX protocol server for Phase 2."""
    
    def __init__(self, host: str = "localhost", port: int = 9878):
        self.host = host
        self.port = port
        self.sessions: Dict[str, FIXSession] = {}
        self.is_running = False
    
    async def start(self):
        """Start the FIX server."""
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        self.is_running = True
        print(f"FIX Server started on {self.host}:{self.port}")
        
        async with self.server:
            await self.server.serve_forever()
    
    async def handle_client(self, reader, writer):
        """Handle incoming FIX client connection."""
        client_addr = writer.get_extra_info('peername')
        print(f"New FIX client connected: {client_addr}")
        
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                
                message = data.decode('utf-8')
                response = await self.process_fix_message(message)
                
                if response:
                    writer.write(response.encode('utf-8'))
                    await writer.drain()
        
        except Exception as e:
            print(f"Error handling FIX client {client_addr}: {e}")
        
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"FIX client {client_addr} disconnected")
    
    async def process_fix_message(self, message: str) -> Optional[str]:
        """Process incoming FIX message."""
        # Parse FIX message fields
        fields = self.parse_fix_message(message)
        
        if not fields:
            return None
        
        msg_type = fields.get('35')  # MsgType field
        
        if msg_type == 'A':  # Logon
            return await self.handle_logon(fields)
        elif msg_type == 'D':  # NewOrderSingle
            return await self.handle_new_order(fields)
        elif msg_type == 'F':  # OrderCancelRequest
            return await self.handle_cancel_order(fields)
        elif msg_type == '0':  # Heartbeat
            return await self.handle_heartbeat(fields)
        
        return None
    
    def parse_fix_message(self, message: str) -> Dict[str, str]:
        """Parse FIX message into field dictionary."""
        fields = {}
        
        # Simple FIX parsing (SOH = |)
        parts = message.strip().split('|')
        
        for part in parts:
            if '=' in part:
                tag, value = part.split('=', 1)
                fields[tag] = value
        
        return fields
    
    async def handle_logon(self, fields: Dict[str, str]) -> str:
        """Handle FIX Logon message."""
        sender_comp_id = fields.get('49', '')
        session_id = f"{sender_comp_id}_{datetime.utcnow().timestamp()}"
        
        session = FIXSession(session_id)
        session.is_active = True
        self.sessions[session_id] = session
        
        # Create Logon response
        response = (
            f"8=FIX.4.4|9=100|35=A|34={session.seq_num_out}|"
            f"49=NASDAQ_SIM|56={sender_comp_id}|52={datetime.utcnow().strftime('%Y%m%d-%H:%M:%S')}|"
            f"98=0|108=30|10=000|"
        )
        
        session.seq_num_out += 1
        return response
    
    async def handle_new_order(self, fields: Dict[str, str]) -> str:
        """Handle New Order Single message."""
        cl_ord_id = fields.get('11', '')
        symbol = fields.get('55', '')
        side = fields.get('54', '')
        order_qty = fields.get('38', '0')
        ord_type = fields.get('40', '')
        price = fields.get('44', '')
        
        # Generate execution report
        exec_id = f"EXEC_{datetime.utcnow().timestamp()}"
        order_id = f"ORD_{datetime.utcnow().timestamp()}"
        
        response = (
            f"8=FIX.4.4|9=200|35=8|34=2|"
            f"49=NASDAQ_SIM|56=CLIENT|52={datetime.utcnow().strftime('%Y%m%d-%H:%M:%S')}|"
            f"37={order_id}|11={cl_ord_id}|17={exec_id}|150=0|39=0|"
            f"55={symbol}|54={side}|38={order_qty}|40={ord_type}|44={price}|"
            f"10=000|"
        )
        
        return response
    
    async def handle_cancel_order(self, fields: Dict[str, str]) -> str:
        """Handle Order Cancel Request."""
        cl_ord_id = fields.get('11', '')
        orig_cl_ord_id = fields.get('41', '')
        
        # Generate cancel confirmation
        exec_id = f"CANCEL_{datetime.utcnow().timestamp()}"
        
        response = (
            f"8=FIX.4.4|9=150|35=8|34=3|"
            f"49=NASDAQ_SIM|56=CLIENT|52={datetime.utcnow().strftime('%Y%m%d-%H:%M:%S')}|"
            f"11={cl_ord_id}|41={orig_cl_ord_id}|17={exec_id}|150=4|39=4|"
            f"10=000|"
        )
        
        return response
    
    async def handle_heartbeat(self, fields: Dict[str, str]) -> str:
        """Handle Heartbeat message."""
        test_req_id = fields.get('112', '')
        
        response = (
            f"8=FIX.4.4|9=80|35=0|34=4|"
            f"49=NASDAQ_SIM|56=CLIENT|52={datetime.utcnow().strftime('%Y%m%d-%H:%M:%S')}|"
        )
        
        if test_req_id:
            response += f"112={test_req_id}|"
        
        response += "10=000|"
        return response


if __name__ == "__main__":
    server = FIXServer()
    asyncio.run(server.start())
