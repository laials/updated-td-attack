#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>

uint64_t virt_to_phys(void *virt) {
    uint64_t value;
    uint64_t page_size = getpagesize();
    uint64_t virt_addr = (uint64_t)virt;
    uint64_t offset = (virt_addr / page_size) * 8;

    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;

    if (pread(fd, &value, 8, offset) != 8) {
        close(fd);
        return 0;
    }

    close(fd);

    if (!(value & (1ULL << 63))) return 0;

    uint64_t pfn = value & ((1ULL << 55) - 1);
    return (pfn * page_size) + (virt_addr % page_size);
}
