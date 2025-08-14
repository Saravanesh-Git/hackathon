document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.querySelector("form");

    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Get form values
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value;

        // Basic validation
        if (!username || !password) {
            alert("Please fill in both username and password.");
            return;
        }

        const payload = {
            username: username,
            password: password
        };

        try {
            const res = await fetch("/api/login", {  // Uses Apache reverse proxy
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            // Handle non-OK HTTP responses
            if (!res.ok) {
                let errMessage = "Login failed!";
                try {
                    const errData = await res.json();
                    errMessage = errData.detail || errMessage;
                } catch (_) {}
                alert(`Error: ${errMessage}`);
                return;
            }

            // Parse success response
            const data = await res.json();
            alert(data.Message || "Login successful!");

            // Redirect to dashboard or desired page
            window.location.href = "home";

        } catch (error) {
            console.error("Login error:", error);
            alert("Unable to connect to the server. Please try again later.");
        }
    });
});
