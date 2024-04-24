document.addEventListener("DOMContentLoaded", function () {
    // Simulating a delay for fetching biography
    setTimeout(() => {
        const bio = `Dashrath Manjhi, also known as the "Mountain Man" of India, was a remarkable individual who dedicated over two decades of his life to carve a path through a mountain using only a chisel and hammer. Born in 1934 in the Gehlaur village of Bihar, India, Manjhi's determination and resilience have inspired people around the world.
            Manjhi's extraordinary feat came from a deep sense of love and devotion to his community. The mountain stood as a barrier between his village and essential resources, making it difficult for villagers to access medical facilities, schools, and markets. Instead of accepting the status quo, Manjhi decided to take matters into his own hands.
            With sheer determination and relentless effort, Manjhi worked tirelessly for 22 years, chiseling away at the rocky terrain to create a road that reduced the travel distance from his village to the nearest town from 55 kilometers to just 15 kilometers. His perseverance and selflessness earned him the admiration of people worldwide.
            Dashrath Manjhi's legacy continues to inspire generations, reminding us that with unwavering resolve and dedication, even the seemingly impossible can be achieved.`;
        document.getElementById("bio").textContent = bio;
    }, 1000); // Delayed for 1 second to simulate loading time
});