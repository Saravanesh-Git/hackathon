document.addEventListener("DOMContentLoaded", () => {
    const signupForm = document.querySelector("form");

    signupForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Get form values
        const username = document.getElementById("username").value.trim();
        const role = document.getElementById("role").value;
        const email = document.getElementById("email").value.trim();
        const mobile = document.getElementById("mobile").value.trim();
        const password = document.getElementById("password").value;
        const confirmPassword = document.getElementById("confirm-password").value;

        // Basic validation before sending to backend
        if (!username || !role || !email || !mobile || !password || !confirmPassword) {
            alert("Please fill all the fields.");
            return;
        }

        if (password !== confirmPassword) {
            alert("Passwords do not match!");
            return;
        }

        const payload = {
            username: username,
            role: role,
            email: email,
            mobile: mobile,
            password: password
        };

        try {
            const res = await fetch("/api/signup", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const err = await res.json();
                alert(`Error: ${err.detail || "Unknown error"}`);
                return;
            }

            const data = await res.json();
            alert(data.Message || "Signup successful!");
            signupForm.reset();
            
            // Optionally redirect to login page
            window.location.href = "login";

        } catch (error) {
            console.error("Signup error:", error);
            alert("Unable to connect to the server.");
        }
    });
});
