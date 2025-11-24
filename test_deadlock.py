"""Test to verify no deadlock in topology"""

from topology import create_topology
import threading
import time

def test_concurrent_access():
    """Test concurrent access to topology methods"""
    print("Testing concurrent access to topology methods...")
    
    topo = create_topology('ring', 10, hops=[1, 2])
    
    def worker1():
        for i in range(100):
            neighbors = topo.get_neighbors(i % 10)
            
    def worker2():
        for i in range(100):
            info = topo.get_topology_info()
            
    def worker3():
        for i in range(100):
            edges = topo.get_all_edges()
    
    threads = [
        threading.Thread(target=worker1),
        threading.Thread(target=worker2),
        threading.Thread(target=worker3)
    ]
    
    start = time.time()
    for t in threads:
        t.start()
    
    for t in threads:
        t.join(timeout=5.0)  # 5 second timeout
    
    elapsed = time.time() - start
    
    # Check if any thread is still alive (deadlock)
    alive = [t for t in threads if t.is_alive()]
    if alive:
        print(f"❌ DEADLOCK DETECTED! {len(alive)} threads still running after 5s")
        return False
    else:
        print(f"✓ No deadlock! Completed in {elapsed:.2f}s")
        return True

def test_all_topologies():
    """Test all topology types"""
    print("\nTesting all topology types for deadlock...")
    
    topologies = [
        ('ring', {'hops': [1]}),
        ('line', {'bidirectional': True}),
        ('mesh', {'edges': [(0,1), (1,2), (2,0)]}),
        ('full', {})
    ]
    
    for topo_type, params in topologies:
        print(f"\n  Testing {topo_type} topology...")
        topo = create_topology(topo_type, 5, **params)
        
        # Try to get info (this was causing deadlock)
        try:
            info = topo.get_topology_info()
            print(f"  ✓ {topo_type}: {info}")
        except Exception as e:
            print(f"  ❌ {topo_type} failed: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("DEADLOCK TEST")
    print("="*60)
    
    success = test_all_topologies()
    
    if success:
        print("\n" + "="*60)
        success = test_concurrent_access()
    
    if success:
        print("\n" + "="*60)
        print("ALL TESTS PASSED - NO DEADLOCK ✓")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("TESTS FAILED ❌")
        print("="*60)
