# str = "Python !!!"
# print(str)

# age = 40
# print(age)

# price=10.1
# print(price)

# is_student = True
# print(is_student)

# x= "10"
# y = int(x)
# print(y*y)

# num1 = 10
# num2 = True
# print(num1 > num2)
# print(num1 > 5 and num2 < 5)
# print(num1 * num2)

# age = 18
# if age >= 18:
#     print("Eligible to vote")

# age = 16
# if age >= 18:
#     print("Major")
# else:
#     print("Minor")

# marks = 85
# if marks >= 90:
#     print("A")
# elif marks>= 75:
#     print("B")
# else:
#     print("C")

# marks = 85
# match marks:
#     case 90:
#         print("A")
#     case 85:
#         print("B")
#     case 70:
#         print("C")
#     case _:
#         print("Fail")

# def test_func():
#     print("No Parameter - No Return Type")

# test_func()

# def test_func():
#     return "No Parameter - With Return Type"

# res = test_func()
# print(res)


# def test_func(param1):
#     print(param1)

# test_func("With Parameter - No Return Type")


# def test_func(param1):
#     return param1

# res = test_func("With Parameter - With Return Type")
# print(res)


# score = {
#     90 : "A Grade",
#     80 : "B Grade",
#     70 : "C Grade",
#     60 : "Fail"
# }
# print(score)
# print(score.get(90))
# print(score.get(40,"Invalid"))


# mathematics = {
#     "+" : lambda num1,num2:num1+num2,
#     "-" : lambda num1,num2:num1-num2,
#     "*" : lambda num1,num2:num1*num2,
#     "/" : lambda num1,num2:num1/num2
# }

# print(mathematics["+"](200,100))

# numbers = [10,20,30,40,50]
# for num in numbers:
#     print(num)

# tuple = (100,200,300)
# print(tuple)
# print(tuple[0])
# print(tuple[-1])
# for tup in tuple:
#     print(tup)


a, *b = (1,2,3,4,5)
print(a)
for num in b:
    print(num)


# 1️⃣ User Entity (ALREADY OK ✅)

You already have:

```java
private String resetToken;
```

✔ No change needed

---

# 2️⃣ UserRepository (ADD 2 METHODS)

```java
package com.careerit.repository;

import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import com.careerit.entity.User;

public interface UserRepository extends JpaRepository<User, Long> {

    Optional<User> findByUsername(String username);

    Optional<User> findByEmail(String email);

    // ✅ ADD
    Optional<User> findByResetToken(String resetToken);
}
```

---

# 3️⃣ Email Utility (NEW FILE)

### 📄 `EmailService.java`

```java
package com.careerit.service;

import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailService {

    private final JavaMailSender mailSender;

    public EmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }

    public void sendResetEmail(String to, String token) {

        String resetLink = "http://localhost:8000/reset-password?token=" + token;

        SimpleMailMessage message = new SimpleMailMessage();
        message.setTo(to);
        message.setSubject("Password Reset");
        message.setText(
            "Click the link to reset password:\n" + resetLink
        );

        mailSender.send(message);
    }
}
```

---

# 4️⃣ AuthService (ADD 2 METHODS ONLY)

```java
package com.careerit.service;

import java.util.Optional;
import java.util.UUID;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.careerit.entity.User;
import com.careerit.repository.UserRepository;

@Service
public class AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final EmailService emailService;

    public AuthService(UserRepository userRepository,
                       PasswordEncoder passwordEncoder,
                       EmailService emailService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.emailService = emailService;
    }

    // ============================
    // FORGOT PASSWORD
    // ============================
    public String forgotPassword(String email) {

        Optional<User> optionalUser = userRepository.findByEmail(email);

        if (optionalUser.isEmpty()) {
            return "Email not registered";
        }

        User user = optionalUser.get();
        String token = UUID.randomUUID().toString();

        user.setResetToken(token);
        userRepository.save(user);

        emailService.sendResetEmail(email, token);
        return "Reset link sent to email";
    }

    // ============================
    // RESET PASSWORD
    // ============================
    public String resetPassword(String token, String newPassword) {

        Optional<User> optionalUser =
                userRepository.findByResetToken(token);

        if (optionalUser.isEmpty()) {
            return "Invalid or expired token";
        }

        User user = optionalUser.get();
        user.setPassword(passwordEncoder.encode(newPassword));
        user.setResetToken(null); // invalidate token

        userRepository.save(user);
        return "Password reset successful";
    }
}
```

---

# 5️⃣ AuthController (ADD 2 ENDPOINTS)

```java
package com.careerit.controller;

import org.springframework.web.bind.annotation.*;

import com.careerit.dto.ForgotPasswordRequest;
import com.careerit.service.AuthService;

@RestController
@CrossOrigin
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    // ============================
    // FORGOT PASSWORD
    // ============================
    @PostMapping("/forget-password")
    public String forgotPassword(@RequestBody ForgotPasswordRequest request) {
        return authService.forgotPassword(request.getEmail());
    }

    // ============================
    // RESET PASSWORD
    // ============================
    @PostMapping("/reset-password")
    public String resetPassword(@RequestParam String token,
                                @RequestParam String newPassword) {
        return authService.resetPassword(token, newPassword);
    }
}
```

---

# 6️⃣ ForgotPasswordRequest DTO (ALREADY OK ✅)

```java
public class ForgotPasswordRequest {
    private String email;
}
```

---

# 7️⃣ application.properties (MAIL CONFIG)

```properties
spring.mail.host=smtp.gmail.com
spring.mail.port=587
spring.mail.username=your_email@gmail.com
spring.mail.password=APP_PASSWORD
spring.mail.properties.mail.smtp.auth=true
spring.mail.properties.mail.smtp.starttls.enable=true
```

📌 Use **Gmail App Password**, NOT normal password.

---

# 8️⃣ SecurityConfig (ALREADY OK ✅)

You already have:

```java
.requestMatchers("/register", "/login", "/forget-password", "/reset-password").permitAll()
```

✔ No change needed

---

# ✅ API TESTING

### 🔹 Forgot Password

```http
POST /forget-password
{
  "email": "test@gmail.com"
}
```

📧 Email sent with link.

---

### 🔹 Reset Password

```http
POST /reset-password?token=XXXX&newPassword=123456
```

